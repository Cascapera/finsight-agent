"""Suite de avaliação de RAG (métricas no estilo RAGAS, implementadas localmente)."""

from finsight.evals.dataset import (
    SEED_DATASET,
    SEED_SAMPLES,
    EvalDataset,
    EvalSample,
)
from finsight.evals.generator import generate_answer

__all__ = [
    "SEED_DATASET",
    "SEED_SAMPLES",
    "EvalDataset",
    "EvalSample",
    "generate_answer",
]
