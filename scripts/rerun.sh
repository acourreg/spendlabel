#!/usr/bin/env bash
#
# Re-run a SINGLE consumer paradigm from scratch (one at a time, never all at once):
#   1. wipe its rows in postgres (classifications + metrics)
#   2. reset its Kafka consumer-group offset on cpv-raw to earliest
#   3. launch ONLY that paradigm's consumer job (foreground; exits when idle)
#   4. score the run (ground_truth + accuracy) -> metrics -> ui-chart shows it
#
# Usage: bash scripts/rerun.sh <paradigm>

set -euo pipefail

VALID_PARADIGMS="hardcoded spark_ml deeplearning_onnx solver langchain n8n mcp"

usage() {
  echo "Usage: bash scripts/rerun.sh <paradigm>" >&2
  echo "Valid paradigms: ${VALID_PARADIGMS}" >&2
}

if [ "$#" -lt 1 ]; then
  echo "Error: missing paradigm argument." >&2
  usage
  exit 1
fi

PARADIGM="$1"

case " ${VALID_PARADIGMS} " in
  *" ${PARADIGM} "*) ;;
  *)
    echo "Error: unknown paradigm '${PARADIGM}'." >&2
    usage
    exit 1
    ;;
esac

# run from the project root no matter where the script is invoked from
cd "$(dirname "$0")/.."

echo ">> [1/4] Deleting '${PARADIGM}' rows from postgres (classifications + metrics)..."
docker compose exec -T postgres \
  psql -U postgres spendlabel -c \
  "DELETE FROM classifications WHERE paradigm='$PARADIGM'; DELETE FROM metrics WHERE paradigm='$PARADIGM';"

echo ">> [2/4] Resetting consumer group 'spendlabel-${PARADIGM}' offsets on cpv-raw to earliest..."
docker compose exec -T kafka \
  kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group "spendlabel-$PARADIGM" --topic cpv-raw \
  --reset-offsets --to-earliest --execute

echo ">> [3/4] Running ONLY the '${PARADIGM}' consumer job (foreground; exits when idle)..."
# A one-off container for just this paradigm — not the others. Blocks until the
# job idle-stops, hits the record cap, or hits the time cap; streams its logs.
# Forward run caps to the container if set in the environment:
#   MAX_RUN_RECORDS=5000 bash scripts/rerun.sh spark_ml
docker compose run --rm \
  ${MAX_RUN_RECORDS:+-e MAX_RUN_RECORDS} \
  ${MAX_RUN_SECONDS:+-e MAX_RUN_SECONDS} \
  consumers python main.py --paradigm "$PARADIGM"

echo ">> [4/4] Scoring the run (ground truth + accuracy -> metrics)..."
# The ONLY step that reads the dataset's ground truth; fills is_correct per row and
# materialises the metrics aggregate the ui-chart reads. Needs psycopg2 on the host;
# override PYTHON to point at a venv that has it, e.g.
#   PYTHON=services/flink-processor/.venv/bin/python bash scripts/rerun.sh spark_ml
"${PYTHON:-python3}" services/flink-processor/app/consumers/accuracy.py --paradigm "$PARADIGM"

echo ">> done — results for '${PARADIGM}' are live in the dashboard (http://localhost:8080)."
