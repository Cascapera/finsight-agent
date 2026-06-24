"""
Testes do runner (Passo 4 da Semana 4) — sem banco, sem rede.

A chave de testabilidade do runner é a estratégia INJETÁVEL: passamos uma
estratégia-fake que devolve RetrievedChunk fabricados (não toca no pgvector) e
monkeypatcham-se `generate_answer` e as 4 métricas para retornar valores
determinísticos. Assim exercitamos a ORQUESTRAÇÃO (recupera -> gera -> mede) e a
AGREGAÇÃO sem nenhuma dependência externa.

`aggregate` e `format_report` são funções puras — testadas com SampleResult
fabricados à mão, sem mock algum.
"""

import uuid
from typing import Any

import pytest

from finsight.evals import runner
from finsight.evals.dataset import EvalDataset, EvalSample
from finsight.evals.metrics import MetricResult
from finsight.evals.runner import (
    SampleResult,
    aggregate,
    evaluate_sample,
    format_report,
    run_evaluation,
)
from finsight.retrieval.retriever import RetrievedChunk


def _chunk(content: str, score: float = 0.9) -> RetrievedChunk:
    """RetrievedChunk mínimo para os fakes — só content/score importam aqui."""
    return RetrievedChunk(
        content=content,
        score=score,
        document_id=uuid.uuid4(),
        document_title="doc",
        chunk_index=0,
    )


def _sample(sample_id: str = "q1") -> EvalSample:
    return EvalSample(
        id=sample_id,
        question="Qual a receita?",
        ground_truth="A receita foi R$ 48,2 bi.",
        ground_truth_contexts=["A receita líquida atingiu R$ 48,2 bi."],
        ticker="PNOR3",
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scores: dict[str, float],
    answer: str = "resposta gerada",
) -> None:
    """Fixa generate_answer e as 4 métricas em valores determinísticos."""

    async def fake_generate(question: str, contexts: list[str]) -> str:
        return answer

    monkeypatch.setattr(runner, "generate_answer", fake_generate)

    def make_metric(name: str) -> Any:
        async def metric(*args: Any, **kwargs: Any) -> MetricResult:
            return MetricResult(scores[name])

        return metric

    monkeypatch.setattr(runner, "faithfulness", make_metric("faithfulness"))
    monkeypatch.setattr(runner, "answer_relevancy", make_metric("answer_relevancy"))
    monkeypatch.setattr(runner, "context_precision", make_metric("context_precision"))
    monkeypatch.setattr(runner, "context_recall", make_metric("context_recall"))


# ===========================================================================
# evaluate_sample — orquestração de um sample
# ===========================================================================


@pytest.mark.asyncio
async def test_evaluate_sample_runs_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A estratégia injetada recebe a pergunta+ticker e os 4 scores chegam ao resultado."""
    captured: dict[str, Any] = {}

    async def fake_strategy(
        query: str, *, ticker: str | None, session: Any
    ) -> list[RetrievedChunk]:
        captured["query"] = query
        captured["ticker"] = ticker
        return [_chunk("contexto recuperado")]

    scores = {
        "faithfulness": 0.8,
        "answer_relevancy": 0.7,
        "context_precision": 0.6,
        "context_recall": 0.5,
    }
    _patch_pipeline(monkeypatch, scores=scores, answer="A receita foi alta.")

    result = await evaluate_sample(_sample(), fake_strategy, strategy_name="baseline", session=None)

    assert captured["query"] == "Qual a receita?"
    assert captured["ticker"] == "PNOR3"  # o ticker do sample é repassado à estratégia
    assert result.sample_id == "q1"
    assert result.strategy == "baseline"
    assert result.answer == "A receita foi alta."
    assert result.scores == scores


# ===========================================================================
# run_evaluation — produto cartesiano estratégia x sample
# ===========================================================================


@pytest.mark.asyncio
async def test_run_evaluation_covers_all_strategies_and_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2 estratégias x 2 samples -> 4 SampleResult, cada par presente uma vez."""

    async def strat_a(query: str, *, ticker: str | None, session: Any) -> list[RetrievedChunk]:
        return [_chunk("ctx a")]

    async def strat_b(query: str, *, ticker: str | None, session: Any) -> list[RetrievedChunk]:
        return [_chunk("ctx b")]

    _patch_pipeline(
        monkeypatch,
        scores={
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
        },
    )

    dataset = EvalDataset(samples=[_sample("q1"), _sample("q2")])
    results = await run_evaluation(dataset, {"strat_a": strat_a, "strat_b": strat_b}, session=None)

    assert len(results) == 4
    pairs = {(r.strategy, r.sample_id) for r in results}
    assert pairs == {
        ("strat_a", "q1"),
        ("strat_a", "q2"),
        ("strat_b", "q1"),
        ("strat_b", "q2"),
    }


# ===========================================================================
# aggregate / format_report — funções puras
# ===========================================================================


def test_aggregate_averages_per_strategy() -> None:
    """Médias por (estratégia, métrica) sobre os samples; n_samples correto."""
    results = [
        SampleResult("q1", "baseline", dict.fromkeys(runner.METRIC_NAMES, 0.4)),
        SampleResult("q2", "baseline", dict.fromkeys(runner.METRIC_NAMES, 0.6)),
        SampleResult("q1", "rerank", dict.fromkeys(runner.METRIC_NAMES, 1.0)),
    ]
    reports = aggregate(results)

    assert reports["baseline"].n_samples == 2
    assert reports["baseline"].means["faithfulness"] == pytest.approx(0.5)
    assert reports["rerank"].n_samples == 1
    assert reports["rerank"].means["context_recall"] == pytest.approx(1.0)


def test_aggregate_preserves_strategy_order() -> None:
    """A ordem de primeira aparição da estratégia é preservada (dict insertion order)."""
    results = [
        SampleResult("q1", "baseline", dict.fromkeys(runner.METRIC_NAMES, 0.5)),
        SampleResult("q1", "hyde", dict.fromkeys(runner.METRIC_NAMES, 0.5)),
        SampleResult("q1", "rerank", dict.fromkeys(runner.METRIC_NAMES, 0.5)),
    ]
    assert list(aggregate(results).keys()) == ["baseline", "hyde", "rerank"]


def test_format_report_is_markdown_table() -> None:
    """A tabela tem cabeçalho com as 4 métricas e uma linha por estratégia."""
    results = [
        SampleResult("q1", "baseline", dict.fromkeys(runner.METRIC_NAMES, 0.5)),
        SampleResult("q1", "rerank", dict.fromkeys(runner.METRIC_NAMES, 0.8)),
    ]
    report = format_report(results)
    lines = report.splitlines()

    # Cabeçalho carrega todas as métricas, na ordem canônica.
    for metric in runner.METRIC_NAMES:
        assert metric in lines[0]
    # Uma linha de dados por estratégia, com os valores formatados.
    assert any(line.startswith("| baseline |") and "0.500" in line for line in lines)
    assert any(line.startswith("| rerank |") and "0.800" in line for line in lines)
