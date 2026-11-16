// free-model-aggregator-02: one server, two client wires, and the best live free model behind
// every answer.
//
// This entry keeps the starter kit's shape (lib/providers.ts reads the brief and builds the
// catalog; lib/wire.ts translates the Anthropic wire both ways and keeps the `model` field honest)
// and replaces its routing and its failover with the two things the score pays for:
//
//   - quality (60%): every free model that meets a prompt's requirements is ranked by the brief's
//     rating_milli, best first, and the prompt is answered by the best model that will take it.
//     JSON-mode prompts go to the best JSON-capable free model (nemotron-3.5-super-vl, 1303);
//     plain and tools prompts go to the best free model overall (inkling-2-small, 1475).
//   - failover (10%): a gateway call that comes back 4xx/5xx, or breaks in flight, or answers
//     with no text at all, is retried on the next best model that meets the prompt, up to five
//     attempts inside the client's 30 s timeout. Every refusal is recorded on the response's
//     `x-makefun-upstream-failures` header, the only way the failover column sees a refusal that
//     was retried and then served. A model that refuses is cooled down per model (60 s on a 429,
//     30 s on a 5xx or a network break), so one rate-limited model never takes the later prompts
//     down with it, and a cooling model is only dialled again when nothing fresh is left.
//   - the catalog (5%): exactly the brief's fourteen free ids, in the register body and in both
//     answers to GET /v1/models, byte for byte the same document `node main.ts --list` prints.
//
// A trial runs this server as `node /bot/main.ts` from the repository's root, with Node 22's
// standard library alone (nothing is installed; TypeScript runs directly). The env is at
// $MAKEFUN_ENV_URL; model calls go through the grader's gateway at $OPENAI_BASE_URL, an
// OpenAI-compatible endpoint that reaches OpenRouter on the key pushed to the competition.
// OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for the entrant's own key; the real
// key never enters the sandbox, and this server never writes it anywhere.
//
// The server:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//      Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages,
//      each with a JSON body or with server-sent events;
//   4. answers each prompt by dialling a free model through the gateway, best first, failing over
//      on a refusal, and reporting the model it actually dialled in every response's `model` field;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//      A process still running when the wall ends scores zero, so never wait for anything else.
//
// Determinism: no clock is read for any document the trial scores (the model lists use the fixed
// timestamps in lib/wire.ts, and the catalog is built once from the brief). The only clock reads
// are the per-prompt deadline and the per-model cooldowns, which shape which model is dialled, not
// the bytes of any scored document, so a live pass and its replay agree.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most upstream attempts spent on one prompt: the best model, then the next best, and no more. */
const MAX_ATTEMPTS = 5;
/** One prompt's whole budget: the client times out at 30 s, so answer inside 28 s. */
const PROMPT_BUDGET_MS = 28000;
/** The first attempt's own cap: a hung model must not eat the whole budget. */
const FIRST_ATTEMPT_MS = 15000;
/** The least time left that is still worth another attempt. */
const MIN_REMAINING_MS = 5000;
/** A refused model is not dialled again while it cools down. */
const COOLDOWN_429_MS = 60000;
const COOLDOWN_5XX_MS = 30000;
const COOLDOWN_NET_MS = 30000;
/** The failures header's name and its row cap, as the house reads them. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';
const MAX_FAILURES = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Per-model cooldown: the epoch millisecond a refused model may be dialled again. */
const cooldownUntil = new Map<string, number>();

/** One upstream refusal, as the failures header carries it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Every free model that meets the requirements, best first: the brief's rating_milli, then the
 * longest context, then the id, so the order is deterministic and the best model is always first.
 * Quality is 60% of the score, so this order is the whole game.
 */
function rankCandidates(requires: Requirement): ModelRow[] {
  const out: ModelRow[] = [];
  for (const model of freeModels(brief)) {
    if (meetsRequirements(requires, capabilitiesOf(model))) out.push(model);
  }
  out.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    const ca = a.context_length ?? 0;
    const cb = b.context_length ?? 0;
    if (cb !== ca) return cb - ca;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return out;
}

/**
 * The dial order for one prompt: models that are not cooling down first (best first), then the
 * cooling ones by the earliest expiry, so a cooling model is only dialled when nothing fresh is left.
 */
function dialOrder(candidates: ModelRow[]): ModelRow[] {
  const now = Date.now();
  const fresh = candidates.filter((m) => (cooldownUntil.get(m.id) ?? 0) <= now);
  const cooling = candidates
    .filter((m) => (cooldownUntil.get(m.id) ?? 0) > now)
    .sort((a, b) => (cooldownUntil.get(a.id) ?? 0) - (cooldownUntil.get(b.id) ?? 0));
  return [...fresh, ...cooling];
}

/** Cool a refused model down: a 429 or a 403 longer than a 5xx, a network break like a 5xx, a 4xx not at all. A 403 is a gate on the model (the top-rated free model is only served to agentic harnesses), so it stays down for the rest of the trial rather than being retried on every prompt. */
function coolDown(model: string, status: number | null, network: boolean): void {
  const ms = status === 429 || status === 403 ? COOLDOWN_429_MS : status !== null && status >= 500 ? COOLDOWN_5XX_MS : network ? COOLDOWN_NET_MS : 0;
  if (ms > 0) cooldownUntil.set(model, Date.now() + ms);
}

/** The error code a gateway refusal body names, or a label for the HTTP status when it names none. */
function errorCodeOf(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status label.
  }
  const label: Record<number, string> = {
    400: 'bad_request',
    401: 'unauthorized',
    403: 'forbidden',
    404: 'not_found',
    429: 'rate_limited',
    500: 'server_error',
    502: 'bad_gateway',
    503: 'unavailable',
    504: 'timeout',
  };
  return label[status] ?? `http_${status}`;
}

/** One gateway call's outcome: the live response when it answered, else the refusal that ended it. */
interface DialOutcome {
  response: Response | null;
  status: number | null;
  code: string;
  network: boolean;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dialOnce(model: string, body: Json, timeoutMs: number): Promise<DialOutcome> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let response: Response;
  try {
    response = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model }),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    const name = (err as Error).name;
    return { response: null, status: null, code: name === 'TimeoutError' ? 'timeout' : 'unreachable', network: true };
  }
  if (!response.ok || response.body === null) {
    const text = await response.text().catch(() => '');
    return { response: null, status: response.status, code: errorCodeOf(text, response.status), network: false };
  }
  return { response, status: response.status, code: '', network: false };
}

/** The text of a Chat Completions answer's message, as the client reads it for the prompt's token. */
function answerTextOf(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content = message.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  }
  return '';
}

/** One prompt's serving: the model that answered and its gateway answer, or none, with the trail. */
interface ServeResult {
  served: boolean;
  /** The id of the model that answered, '' when none did. */
  model: string;
  /** Streaming: the live gateway response. JSON: null, the answer is parsed in `answer`. */
  response: Response | null;
  /** JSON: the parsed gateway answer. Streaming: null. */
  answer: Json | null;
  /** Every refusal this prompt cost, in the order it cost them. */
  failures: FailureRow[];
}

/**
 * Serve one prompt through the gateway, best model first, failing over on a refusal. A JSON answer
 * with no text at all can never carry the prompt's token, so it counts as a failed attempt too and
 * the next model is tried. A streaming answer cannot be checked before it is sent, so a streaming
 * dial that comes back 2xx is served as it arrives.
 */
async function serve(body: Json, candidates: ModelRow[], deadline: number): Promise<ServeResult> {
  const failures: FailureRow[] = [];
  const order = dialOrder(candidates).slice(0, MAX_ATTEMPTS);
  let first = true;
  for (const model of order) {
    const remaining = deadline - Date.now();
    if (remaining < MIN_REMAINING_MS) break;
    const timeoutMs = first ? Math.min(FIRST_ATTEMPT_MS, remaining) : remaining;
    first = false;
    const outcome = await dialOnce(model.id, body, timeoutMs);
    if (outcome.response === null) {
      failures.push({ status: outcome.status ?? 0, code: outcome.code, model: model.id });
      coolDown(model.id, outcome.status, outcome.network);
      if (failures.length >= MAX_FAILURES) break;
      continue;
    }
    if (body.stream === true) {
      return { served: true, model: model.id, response: outcome.response, answer: null, failures };
    }
    let answer: unknown;
    try {
      answer = await outcome.response.json();
    } catch {
      failures.push({ status: 200, code: 'bad_json', model: model.id });
      if (failures.length >= MAX_FAILURES) break;
      continue;
    }
    if (!isRecord(answer)) {
      failures.push({ status: 200, code: 'bad_json', model: model.id });
      if (failures.length >= MAX_FAILURES) break;
      continue;
    }
    if (answerTextOf(answer) === '') {
      failures.push({ status: 200, code: 'empty_response', model: model.id });
      if (failures.length >= MAX_FAILURES) break;
      continue;
    }
    return { served: true, model: model.id, response: null, answer, failures };
  }
  return { served: false, model: '', response: null, answer: null, failures };
}

/** The failures header for a response, empty when the prompt cost no refusal. */
function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { [FAILURES_HEADER]: JSON.stringify(failures.slice(0, MAX_FAILURES)) };
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, plus the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBrief: Requirement = entry !== undefined ? entry.requires : {};
  const fromBody = requirementsFromBody(body);
  // The body's own requirements are ground truth (a JSON-mode body needs a JSON model, a body with
  // tools needs a tools model); the brief's row is the declared source. A model must meet both.
  return {
    ...(fromBrief.tools === true || fromBody.tools === true ? { tools: true } : {}),
    ...(fromBrief.vision === true || fromBody.vision === true ? { vision: true } : {}),
    ...(fromBrief.json === true || fromBody.json === true ? { json: true } : {}),
    ...(fromBrief.min_context !== undefined ? { min_context: fromBrief.min_context } : {}),
    ...(fromBody.min_context !== undefined ? { min_context: fromBody.min_context } : {}),
  };
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extra: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extra);
  else sendJson(res, status, openAIError(message, type, null), extra);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that takes it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankCandidates(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  // Ask the gateway for usage on a stream: the wire's usage check needs it, and a client body that
  // forgot stream_options still gets one.
  const upstreamBody: Json = { ...body };
  if (upstreamBody.stream === true && !isRecord(upstreamBody.stream_options)) {
    upstreamBody.stream_options = { include_usage: true };
  }
  const served = await serve(upstreamBody, candidates, Date.now() + PROMPT_BUDGET_MS);
  const trail = failuresHeader(served.failures);
  if (!served.served || served.model === '') {
    log(`chat: no free model answered (${served.failures.length} refusal(s))`);
    return sendError(res, 'openai', 502, 'upstream_error', 'no free model answered this prompt', trail);
  }
  log(`chat: ${served.model} answered${served.failures.length > 0 ? ` after ${served.failures.length} refusal(s)` : ''}`);
  if (upstreamBody.stream === true && served.response !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const frame of openAIStreamWithModel(served.response.body, served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  if (served.answer !== null) {
    sendJson(res, 200, openAIJsonWithModel(served.answer, served.model), trail);
    return;
  }
  sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered without a body', trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankCandidates(requirementsFor(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  await serveAnthropic(res, body, candidates);
}

/**
 * POST /v1/messages, served properly: the Anthropic request is rewritten to Chat Completions once
 * per attempt, because the upstream body must name the model that attempt dials. The failover loop
 * mirrors serve() but rewrites the body per attempt.
 */
async function serveAnthropic(res: ServerResponse, body: Json, candidates: ModelRow[]): Promise<void> {
  const deadline = Date.now() + PROMPT_BUDGET_MS;
  const failures: FailureRow[] = [];
  const order = dialOrder(candidates).slice(0, MAX_ATTEMPTS);
  let first = true;
  for (const model of order) {
    const remaining = deadline - Date.now();
    if (remaining < MIN_REMAINING_MS) break;
    const timeoutMs = first ? Math.min(FIRST_ATTEMPT_MS, remaining) : remaining;
    first = false;
    const upstreamBody = anthropicToOpenAI(body, model.id);
    const outcome = await dialOnce(model.id, upstreamBody, timeoutMs);
    if (outcome.response === null) {
      failures.push({ status: outcome.status ?? 0, code: outcome.code, model: model.id });
      coolDown(model.id, outcome.status, outcome.network);
      if (failures.length >= MAX_FAILURES) break;
      continue;
    }
    const trail = failuresHeader(failures);
    if (body.stream === true) {
      log(`messages: ${model.id} answered${failures.length > 0 ? ` after ${failures.length} refusal(s)` : ''}`);
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
      try {
        for await (const event of anthropicEventsFromOpenAI(outcome.response.body, model.id)) res.write(event);
      } finally {
        res.end();
      }
      return;
    }
    let answer: unknown;
    try {
      answer = await outcome.response.json();
    } catch {
      failures.push({ status: 200, code: 'bad_json', model: model.id });
      if (failures.length >= MAX_FAILURES) break;
      continue;
    }
    if (!isRecord(answer)) {
      failures.push({ status: 200, code: 'bad_json', model: model.id });
      if (failures.length >= MAX_FAILURES) break;
      continue;
    }
    if (answerTextOf(answer) === '') {
      failures.push({ status: 200, code: 'empty_response', model: model.id });
      if (failures.length >= MAX_FAILURES) break;
      continue;
    }
    log(`messages: ${model.id} answered${failures.length > 0 ? ` after ${failures.length} refusal(s)` : ''}`);
    sendJson(res, 200, openAIToAnthropic(answer, model.id), trail);
    return;
  }
  const trail = failuresHeader(failures);
  log(`messages: no free model answered (${failures.length} refusal(s))`);
  sendError(res, 'anthropic', 502, 'api_error', 'no free model answered this prompt', trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
