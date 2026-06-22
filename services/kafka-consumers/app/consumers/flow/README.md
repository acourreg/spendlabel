# `flow` — workflow automation platform (n8n + Gemini)

The **low-code orchestration** paradigm. Classification is delegated to an
**n8n** workflow over a webhook; inside n8n a **Google Gemini** node does the
actual classification. Same model as `mcp` / `langchain` — what differs is that a
visual, citizen-developer workflow drives it instead of Python.

(Formerly named `n8n`; renamed to `flow` to name the *paradigm* — "workflow
automation platform" — rather than the specific tool.)

## Workflow (n8n)

```
Webhook (POST /cpv-classify)
   → Google Gemini node  (prompt: title + description → 2-digit CPV division)
   → Respond to Webhook  ({ "cpv": "45" })
```

A ready-to-import workflow is in [`workflow.json`](workflow.json). Import it into
your n8n instance, set Gemini credentials on the model node, then activate it.

The Gemini prompt should pin the output to a bare 2-digit code and list the
allowed divisions (mirror `../mcp/cpv_catalogue.py` so all paradigms classify
into the same closed set).

## Setup

1. Import `workflow.json` into n8n; add your **Google Gemini** credential to the
   model node (model `gemini-2.5-flash-lite`). The key lives in n8n, **not** in
   this repo.
2. Activate the workflow and copy its production webhook URL.
3. Point the consumer at it:

```bash
export FLOW_WEBHOOK_URL=https://<your-n8n-host>/webhook/cpv-classify
export FLOW_CONCURRENCY=8     # optional, parallel requests per batch
export FLOW_TIMEOUT=30        # optional, per-request seconds
```

## Request / response contract

The consumer POSTs:
```json
{ "title": "...", "description": "..." }
```
and expects either `{ "cpv": "45" }` (or `predicted_cpv`), or a bare `45`. Any
2-digit token in the reply is accepted; an error or unparseable reply falls back
to `45` (majority class), keeping the paradigm always-predicting.

## Running

> The Gemini calls happen inside n8n and cost money — run on a bounded slice.

```bash
export MAX_RUN_RECORDS=500
python main.py --paradigm flow
```
