# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/DjavaProductions/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/DjavaProductions/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two client wires, and the best live free model behind every answer.

This is an entry for make.fun's **"The best live free-model aggregator"**
(`free-model-aggregator-02`). It serves the OpenAI Chat Completions API and the
Anthropic Messages API — JSON and server-sent events — from a single process,
routing each request to a live OpenRouter free model through the grader's gateway,
and answering with a real completion from that model.

## Run it

Node 22, standard library only. Nothing is installed.

```sh
node main.ts
```

The trial harness sets four environment variables:

| Variable | Meaning |
|---|---|
| `MAKEFUN_ENV_URL` | The trial env (`/register`, `/work`) |
| `MAKEFUN_PROMPT_FILE` | The brief: the free model list and the six prompts |
| `OPENAI_BASE_URL` | The grader's gateway (OpenAI-compatible) |
| `OPENAI_API_KEY` | The gateway lane key (a dummy the gateway swaps for your key) |

Print the catalog the server registers, without starting it:

```sh
node main.ts --list
```

The server listens on `0.0.0.0` at a free port, POSTs
`{protocol_version: 1, port, catalog}` to `$MAKEFUN_ENV_URL/register`, serves
`GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/messages`, polls
`GET $MAKEFUN_ENV_URL/work` every 100 ms, and exits 0 when the env says done.

## How it routes

The brief lists fourteen OpenRouter free models with a published rating
(`rating_milli`) and what each can do (tools, vision, JSON mode, context length).
For every request the server:

1. **Ranks** every free model that meets the request's requirements by rating,
   best first (then longest context, then id, so the order is deterministic).
   JSON-mode prompts go to the best JSON-capable free model; plain and tools
   prompts go to the best free model overall. Quality is 60% of the score, so
   the best model always goes first.
2. **Dials** the best model through the gateway. On a 4xx/5xx, a network break,
   or an answer with no text at all, it **fails over** to the next best model
   that meets the request, up to five attempts inside the client's 30 s timeout
   (28 s budget, 15 s cap on the first attempt). Every refusal is recorded on
   the response's `x-makefun-upstream-failures` header — the only way the
   failover column sees a refusal that was retried and then served.
3. **Cools down** a refused model per model: 60 s on a 429 or a 403, 30 s on a
   5xx or a network break. A model that refuses is not dialled again while it
   cools, so one rate-limited model never takes the later prompts down with it.
   A cooling model is only dialled when nothing fresh is left.
4. **Reports** the model it actually dialled in every response's `model` field
   — the house scores the model that really answered, so the claim is the call
   it made, never another.

The catalog is exactly the brief's fourteen free ids, in the register body and
in both answers to `GET /v1/models` (OpenAI shape on an `Authorization` header,
Anthropic shape on an `x-api-key` header), byte for byte the document
`node main.ts --list` prints.

Two wire details the starter did not handle, found by testing against live
OpenRouter:

- **A provider that sends usage on a stream may send `finish_reason` twice**
  (once on the finish chunk, again on the usage chunk). The wire allows a
  single finish chunk, so the relay strips a second `finish_reason` to null
  and keeps the chunk's usage. Without this, the OpenAI streaming prompt fails
  its stream-order check.
- **The top-rated free model is gated**: `thinkingmachines/inkling-2-small:free`
  (1475) answers 403 unless the request carries an agentic-harness referer,
  and the gateway forwards no referer, so it refuses through the gateway. The
  failover treats a 403 like a 429 (60 s cooldown), so the prompt is served by
  the next best model instead of being lost.

## Files

| File | What it is |
|---|---|
| `main.ts` | The server: the brief, the catalog, the routes, the register and the poll, plus the rating-ranked routing and the failover with per-model cooldowns. |
| `lib/providers.ts` | The brief's types and parser, the free model list, the catalog. (From the starter kit, unchanged.) |
| `lib/wire.ts` | Anthropic ↔ Chat Completions translation, the model field, the SSE writers — plus the duplicate-finish strip above. |
| `LICENSE` | MIT. |

## Tested

Against the kit's real house client (`program/env/server.mjs`) driving the
practice trials, with a local stand-in gateway forwarding to live OpenRouter
free models on a free-tier key:

- **Practice trial-0 and trial-1: 6 of 6 prompts valid, 39 of 39 checks passed**
  (shape, stream order, content token, model, usage, stop reason, tools), on
  both wires, JSON and streaming.
- **Failover verified live**: the top-rated model answers 403, the prompt is
  retried on the next best model and served, and the refusal is recorded on
  the `x-makefun-upstream-failures` header (`failover_handled: 1`).
- **Catalog verified**: 14 models in the register body and in both
  `/v1/models` answers.
- All six prompts were answered by `nvidia/nemotron-3.5-super-vl:free`
  (1303, the best model that answers through the gateway), so the expected
  quality is ~91% of the sealed best — the only loss is the gated top model.

The official `grade-local` path could not run in this sandbox: its relay opens
a direct TLS connection to OpenRouter, and this sandbox only reaches the
internet through its egress proxy, so the relay fails with `EPROTO` before any
model is called. The direct test above drives the same house client and the
same live models, and the server is unchanged between the two paths.
