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
from finsight.evals.runner import (
    DEFAULT_STRATEGIES,
    SampleResult,
    StrategyReport,
    aggregate,
    evaluate_sample,
    format_report,
    run_evaluation,
)

__all__ = [
    "DEFAULT_STRATEGIES",
    "SEED_DATASET",
    "SEED_SAMPLES",
    "EvalDataset",
    "EvalSample",
    "MetricResult",
    "SampleResult",
    "StrategyReport",
    "aggregate",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "evaluate_sample",
    "faithfulness",
    "format_report",
    "generate_answer",
    "run_evaluation",
]
