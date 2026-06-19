-- spendlabel — consumers service schema
-- Created once on startup (idempotent via IF NOT EXISTS).

-- One row per Kafka message processed by a consumer paradigm.
CREATE TABLE IF NOT EXISTS classifications (
  id            SERIAL PRIMARY KEY,
  ocid          TEXT NOT NULL,
  paradigm      TEXT NOT NULL,  -- 'hardcoded' | 'spark_ml' | 'kstreams_onnx' | 'solver' | 'langchain' | 'n8n' | 'mcp'
  predicted_cpv TEXT,           -- code CPV prédit (2 digits)
  ground_truth  TEXT,           -- code CPV réel du dataset
  is_correct    BOOLEAN,        -- predicted == ground_truth
  latency_ms    FLOAT,          -- temps de traitement du message
  value_gbp     FLOAT,          -- montant du contrat
  supplier_name TEXT,
  processed_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_classifications_paradigm ON classifications (paradigm);

-- Aggregate per paradigm (materialised after each run).
CREATE TABLE IF NOT EXISTS metrics (
  paradigm      TEXT PRIMARY KEY,
  accuracy      FLOAT,          -- % is_correct
  avg_latency   FLOAT,
  total_records INT,
  run_at        TIMESTAMPTZ DEFAULT NOW()
);
