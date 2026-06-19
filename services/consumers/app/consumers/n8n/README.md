# n8n Workflow — CPV Classifier

This consumer delegates classification to an **n8n** workflow via webhook.

## Setup

1. Import the workflow JSON (TODO: export and commit) into your n8n instance.
2. The workflow should expose a **Webhook** trigger node at `POST /cpv-classify`.
3. Set the webhook URL in your environment:

```bash
export N8N_WEBHOOK_URL=https://your-n8n-instance.app.n8n.cloud/webhook/cpv-classify
```

## Workflow Design (TODO)

```
Webhook Trigger → Extract description → LLM / Lookup Node → Respond with CPV code
```

## Running

```bash
python main.py --paradigm n8n
```
