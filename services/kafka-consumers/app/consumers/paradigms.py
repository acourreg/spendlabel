"""Single source of truth for the paradigm registry.

Before this, three hand-synced lists (``main.py``, ``accuracy.py``, ``rerun.sh``)
drifted apart — the rename ``n8n -> flow`` and the new ``mcp`` lane left them
inconsistent so ``flow`` could not be scored and ``mcp`` was not dispatched
(audit fix B5). Now ``main.py`` dispatches from ``DISPATCH``, ``accuracy.py``
scores over ``SCOREABLE``, and ``rerun.sh`` reads ``SCOREABLE`` via:

    python -c "from consumers.paradigms import SCOREABLE; print(' '.join(SCOREABLE))"

Kept dependency-free (plain literals) so it imports on the host scorer too.
"""

from __future__ import annotations

# paradigm name -> (module, class). Each class exposes ``run(idle_timeout=...)``.
# Every entry here is fully implemented: it consumes cpv-raw via the shared
# consumer loop, always predicts a code from the canonical CPV set, and is scored
# the same way.
DISPATCH: dict[str, tuple[str, str]] = {
    "hardcoded":                     ("consumers.hardcoded.job", "HardcodedJob"),
    "spark_ml_huggingface":          ("consumers.spark_ml_huggingface.job", "SparkMLConsumer"),
    "deeplearning_onnx_huggingface": ("consumers.deeplearning_onnx_huggingface.job", "DeepLearningJob"),
    "spark_ml_model2vec":            ("consumers.spark_ml_model2vec.job", "Job"),
    "deeplearning_onnx_model2vec":   ("consumers.deeplearning_onnx_model2vec.job", "Job"),
    "solver":                        ("consumers.solver.job", "Job"),
    "langchain":                     ("consumers.langchain.job", "LangChainJob"),
    "mcp":                           ("consumers.mcp.job", "McpJob"),
    "flow":                          ("consumers.flow.job", "FlowJob"),
}

# Order is the dashboard / scoring order. Everything dispatchable is scoreable.
SCOREABLE: list[str] = list(DISPATCH)
