-- spendlabel — demo seed data.
-- Runs after schema.sql (mounted as 02-seed.sql in initdb).
-- ~50 fictional classifications so ui-chart shows real data immediately,
-- then the per-paradigm metrics aggregate is recomputed from those rows.

INSERT INTO classifications
  (ocid, paradigm, predicted_cpv, ground_truth, is_correct, latency_ms, value_gbp, supplier_name)
VALUES
  -- hardcoded — keyword rules: fast, ~57% accurate
  ('ocid-001', 'hardcoded',     '45', '45', true,    7.5, 52000000, 'Balfour Beatty'),
  ('ocid-002', 'hardcoded',     '72', '45', false,   9.1, 47000000, 'Kier Group'),
  ('ocid-003', 'hardcoded',     '72', '72', true,   11.2, 33000000, 'BT Group'),
  ('ocid-004', 'hardcoded',     '48', '72', false,  10.4, 21000000, 'Capita'),
  ('ocid-005', 'hardcoded',     '85', '85', true,    8.9, 64000000, 'NHS Supply Chain'),
  ('ocid-006', 'hardcoded',     '48', '48', true,    6.8,  9000000, 'Sage'),
  ('ocid-007', 'hardcoded',     '71', '90', false,  13.1,  4200000, 'Veolia'),

  -- spark_ml — TF-IDF + classifier: medium latency, ~71% accurate
  ('ocid-008', 'spark_ml',      '45', '45', true,   55.0, 78000000, 'Balfour Beatty'),
  ('ocid-009', 'spark_ml',      '72', '72', true,   61.5, 44000000, 'Fujitsu'),
  ('ocid-010', 'spark_ml',      '79', '72', false,  58.2, 19000000, 'Capita'),
  ('ocid-011', 'spark_ml',      '79', '79', true,   49.0, 16000000, 'Serco'),
  ('ocid-012', 'spark_ml',      '85', '85', true,   72.3, 58000000, 'NHS Supply Chain'),
  ('ocid-013', 'spark_ml',      '48', '48', true,   41.7, 12000000, 'Microsoft UK'),
  ('ocid-014', 'spark_ml',      '73', '71', false,  66.9,  7000000, 'Atkins'),

  -- deeplearning_onnx — neural net via ONNX: low latency, ~86% accurate
  ('ocid-015', 'deeplearning_onnx', '45', '45', true,   22.4, 69000000, 'Skanska'),
  ('ocid-016', 'deeplearning_onnx', '48', '48', true,   19.8, 11000000, 'Oracle UK'),
  ('ocid-017', 'deeplearning_onnx', '60', '60', true,   28.1, 27000000, 'Stagecoach'),
  ('ocid-018', 'deeplearning_onnx', '72', '72', true,   25.6, 51000000, 'IBM UK'),
  ('ocid-019', 'deeplearning_onnx', '79', '79', true,   31.2, 14000000, 'Serco'),
  ('ocid-020', 'deeplearning_onnx', '85', '85', true,   24.9, 60000000, 'NHS Supply Chain'),
  ('ocid-021', 'deeplearning_onnx', '71', '90', false,  33.5,  3800000, 'Veolia'),

  -- solver — OR-Tools constraints: high latency, ~57% accurate
  ('ocid-022', 'solver',        '72', '72', true,   82.0, 38000000, 'Cap Gemini'),
  ('ocid-023', 'solver',        '79', '79', true,   95.5, 22000000, 'Serco'),
  ('ocid-024', 'solver',        '90', '85', false, 110.2, 47000000, 'NHS Supply Chain'),
  ('ocid-025', 'solver',        '90', '90', true,   73.4,  6500000, 'Veolia'),
  ('ocid-026', 'solver',        '75', '98', false,  88.1,  1800000, 'Mitie'),
  ('ocid-027', 'solver',        '75', '75', true,  101.7,  7200000, 'Civica'),
  ('ocid-028', 'solver',        '71', '80', false,  67.9,  9300000, 'Pearson'),

  -- langchain — LLM zero-shot: high latency, ~86% accurate
  ('ocid-029', 'langchain',     '45', '45', true,  540.0, 81000000, 'Balfour Beatty'),
  ('ocid-030', 'langchain',     '48', '48', true,  470.0, 13000000, 'Adobe UK'),
  ('ocid-031', 'langchain',     '72', '72', true,  820.0, 56000000, 'Accenture'),
  ('ocid-032', 'langchain',     '79', '79', true,  610.0, 24000000, 'Serco'),
  ('ocid-033', 'langchain',     '85', '85', true,  760.0, 67000000, 'NHS England'),
  ('ocid-034', 'langchain',     '71', '71', true,  880.0, 17000000, 'Arup'),
  ('ocid-035', 'langchain',     '80', '73', false, 690.0,  4800000, 'Open University'),

  -- n8n — webhook workflow: medium-high latency, ~43% accurate
  ('ocid-036', 'n8n',           '60', '60', true,  320.0, 31000000, 'First Group'),
  ('ocid-037', 'n8n',           '79', '72', false, 280.0, 26000000, 'Capita'),
  ('ocid-038', 'n8n',           '79', '79', true,  410.0, 15000000, 'Serco'),
  ('ocid-039', 'n8n',           '90', '85', false, 450.0, 55000000, 'NHS Supply Chain'),
  ('ocid-040', 'n8n',           '90', '90', true,  210.0,  7400000, 'Veolia'),
  ('ocid-041', 'n8n',           '75', '98', false, 380.0,  2100000, 'Mitie'),
  ('ocid-042', 'n8n',           '72', '45', false, 460.0, 49000000, 'Kier Group'),

  -- mcp — Claude + tools: highest latency, 100% accurate
  ('ocid-043', 'mcp',           '45', '45', true,  980.0, 88000000, 'Balfour Beatty'),
  ('ocid-044', 'mcp',           '48', '48', true,  640.0, 14000000, 'Microsoft UK'),
  ('ocid-045', 'mcp',           '72', '72', true, 1180.0, 62000000, 'Accenture'),
  ('ocid-046', 'mcp',           '79', '79', true,  720.0, 25000000, 'Serco'),
  ('ocid-047', 'mcp',           '85', '85', true,  910.0, 71000000, 'NHS England'),
  ('ocid-048', 'mcp',           '90', '90', true,  680.0,  8900000, 'Veolia'),
  ('ocid-049', 'mcp',           '71', '71', true, 1040.0, 18000000, 'Arup');

-- Recompute the per-paradigm aggregate from the rows just inserted.
INSERT INTO metrics (paradigm, accuracy, avg_latency, total_records, run_at)
SELECT
    paradigm,
    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) AS accuracy,
    AVG(latency_ms)                                 AS avg_latency,
    COUNT(*)                                        AS total_records,
    NOW()                                           AS run_at
FROM classifications
GROUP BY paradigm
ON CONFLICT (paradigm) DO UPDATE SET
    accuracy      = EXCLUDED.accuracy,
    avg_latency   = EXCLUDED.avg_latency,
    total_records = EXCLUDED.total_records,
    run_at        = EXCLUDED.run_at;
