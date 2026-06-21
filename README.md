# spendlabel

**Benchmark: CPV category classification across 7 AI paradigms on real UK public procurement data, streamed via PyFlink + Confluent Cloud Kafka.**

## What It Does

SpendLabel takes real UK government contract notices (from Contracts Finder) and classifies each into one of **15 CPV 2-digit categories** using 7 fundamentally different AI paradigms — then measures accuracy, latency, and throughput head-to-head on the same data stream.

## Architecture

```
CSV files ──► Producer ──► Kafka (cpv-raw) ──► 7 PyFlink consumer jobs ──► PostgreSQL ──► ui-chart dashboard
                                                │
                                                ├── hardcoded     (keyword/regex rules)
                                                ├── spark_ml      (TF-IDF + classifier)
                                                ├── deeplearning_onnx (neural net via ONNX)
                                                ├── solver        (OR-Tools constraints)
                                                ├── langchain     (LLM agent, zero-shot)
                                                ├── n8n           (webhook workflow)
                                                └── mcp           (Claude + tools)
```

Each consumer is a separate PyFlink job subscribed to `cpv-raw`, writing results to its own sink topic. Ground truth (CPV 2-digit prefix) is already in the dataset — no external labelling needed.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Streaming | Apache Flink (PyFlink) |
| Message broker | Confluent Cloud Kafka |
| ML | PySpark, ONNX Runtime |
| Solver | Google OR-Tools |
| LLM | LangChain + OpenAI |
| Workflow | n8n |
| Agentic | MCP + Claude |
| Storage | PostgreSQL |
| Dashboard | Streamlit + Plotly |
| Data | pandas, PyArrow |

## Project Structure

```
spendlabel/
├── data/raw/                       ← 3 CSV files (not committed)
├── services/
│   ├── producer/app/               ← CSV → Kafka publisher (local script, not dockerised)
│   │   └── publish.py
│   ├── flink-processor/app/        ← Service 1: PyFlink jobs → PostgreSQL (dockerised)
│   │   ├── main.py                 ← entrypoint: python main.py --paradigm <p>
│   │   ├── config.py               ← Kafka + Flink settings (env-var overrides)
│   │   ├── consumers/
│   │   │   ├── base_job.py         ← Shared PyFlink job skeleton
│   │   │   ├── hardcoded/          ← Rule-based classifier
│   │   │   ├── spark_ml/           ← Spark ML classifier
│   │   │   ├── deeplearning_onnx/      ← ONNX runtime classifier
│   │   │   ├── solver/             ← Constraint solver classifier
│   │   │   ├── langchain/          ← LangChain LLM classifier
│   │   │   ├── n8n/                ← n8n webhook classifier
│   │   │   └── mcp/                ← MCP + Claude classifier
│   │   ├── db/
│   │   │   ├── schema.sql          ← classifications + metrics tables
│   │   │   └── connection.py       ← psycopg2 pool + insert / materialise
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── ui-chart/app/               ← Service 2: PostgreSQL → charts (Streamlit, dockerised)
│       ├── main.py
│       ├── charts/
│       │   ├── sunburst.py         ← CPV spend sunburst (off-contract colour)
│       │   └── accuracy.py         ← accuracy-per-paradigm table
│       ├── requirements.txt
│       └── Dockerfile
├── config/
│   └── confluent.env.example       ← Kafka + Postgres credentials template
├── docker-compose.yml              ← postgres + consumers + ui-chart
├── notebooks/                      ← EDA on the dataset
├── ATTRIBUTION.md                  ← Data source + licence
├── requirements.txt                ← producer + dev tooling
└── README.md
```

## Quick Start

```bash
# 1. Create virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Kafka credentials
cp config/confluent.env.example config/.env
# Edit config/.env with your Confluent Cloud credentials

# 4. Place CSV files in data/raw/

# 5. Publish data (local script — not containerised)
cd services/producer/app && python publish.py && cd -

# 6. Bring up Postgres + consumers + dashboard
docker compose up --build

# Run a different paradigm:
#   docker compose run --rm consumers python main.py --paradigm spark_ml

# Dashboard: http://localhost:8501
```

## Data

UK Contracts Finder notices via [data.gov.uk](https://data.gov.uk) — Open Government Licence v3.0.
See [ATTRIBUTION.md](ATTRIBUTION.md).
