"""Suite de avaliação de RAG (métricas no estilo RAGAS, implementadas localmente)."""

from finsight.evals.dataset import (
    SEED_DATASET,
    SEED_SAMPLES,
    EvalDataset,
    EvalSample,
)
from finsight.evals.generator import generate_answer
from finsight.evals.metrics import (
    MetricResult,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

__all__ = [
    "SEED_DATASET",
    "SEED_SAMPLES",
    "EvalDataset",
    "EvalSample",
    "MetricResult",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "faithfulness",
    "generate_answer",
]
