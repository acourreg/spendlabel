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

# Valid paradigms come from the single source of truth (audit fix B5).
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)/services/kafka-consumers/app"
VALID_PARADIGMS="$(PYTHONPATH="$APP_DIR" python3 -c 'from consumers.paradigms import SCOREABLE; print(" ".join(SCOREABLE))')"

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

# Fair eval window (audit fixes B2/B3): evaluate on a slice DISJOINT from the ML
# training slice and IDENTICAL for every paradigm. Training uses the first
# TRAIN_POOL records, so eval starts at offset TRAIN_POOL by default and reads the
# next MAX_RUN_RECORDS messages. Override these to move the window — but use the
# SAME values for every paradigm or the numbers are not comparable.
TRAIN_POOL="${TRAIN_POOL:-5000}"
START_OFFSET="${START_OFFSET:-$TRAIN_POOL}"
export MAX_RUN_RECORDS="${MAX_RUN_RECORDS:-5000}"
RESET="--to-offset ${START_OFFSET}"
echo ">> [2/4] Eval window: offset ${START_OFFSET}..$((START_OFFSET + MAX_RUN_RECORDS)) (disjoint from the first ${TRAIN_POOL} training records)"
echo ">> [2/4] Resetting consumer group 'spendlabel-${PARADIGM}' offsets on cpv-raw (${RESET})..."
docker compose exec -T kafka \
  kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group "spendlabel-$PARADIGM" --topic cpv-raw \
  --reset-offsets $RESET --execute

echo ">> [3/4] Running ONLY the '${PARADIGM}' consumer job (foreground; exits when idle)..."
# A one-off container for just this paradigm — not the others. Blocks until the
# job idle-stops, hits the record cap, or hits the time cap; streams its logs.
# Forward run caps to the container if set in the environment:
#   MAX_RUN_RECORDS=5000 bash scripts/rerun.sh spark_ml
docker compose run --rm \
  ${MAX_RUN_RECORDS:+-e MAX_RUN_RECORDS} \
  ${MAX_RUN_SECONDS:+-e MAX_RUN_SECONDS} \
  ${GEMINI_API_KEY:+-e GEMINI_API_KEY} \
  ${FLOW_WEBHOOK_URL:+-e FLOW_WEBHOOK_URL} \
  ${MCP_CONCURRENCY:+-e MCP_CONCURRENCY} \
  ${FLOW_CONCURRENCY:+-e FLOW_CONCURRENCY} \
  consumers python main.py --paradigm "$PARADIGM"

echo ">> [4/4] Scoring the run (ground truth + accuracy -> metrics)..."
# The ONLY step that reads the dataset's ground truth; fills is_correct per row and
# materialises the metrics aggregate the ui-chart reads. Needs psycopg2 on the host;
# override PYTHON to point at a venv that has it, e.g.
#   PYTHON=services/kafka-consumers/.venv/bin/python bash scripts/rerun.sh spark_ml
"${PYTHON:-python3}" services/kafka-consumers/app/consumers/accuracy.py --paradigm "$PARADIGM"

echo ">> done — results for '${PARADIGM}' are live in the dashboard (http://localhost:8080)."
