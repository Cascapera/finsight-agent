"""
Runner — orquestra a avaliação ponta a ponta e compara estratégias de retrieval.

É o Passo 4 (e o fechamento) da eval suite. Os Passos 1-3 deram as PEÇAS:
    - dataset.py   -> o gabarito (golden set)
    - retriever/hyde/reranker (Semana 3) -> as ESTRATÉGIAS de recuperação
    - generator.py -> o "G" (contexto -> resposta)
    - metrics.py   -> as 4 réguas

Este módulo COSTURA tudo: para cada pergunta do golden set e cada estratégia,
recupera -> gera -> mede, e agrega numa tabela comparativa. O objetivo final da
semana é responder, COM NÚMEROS, à pergunta que abriu a Semana 3:

    "HyDE e re-ranking melhoram o RAG? Melhoram O QUÊ, e quanto?"

>>> Por que o runner é orquestração PURA (e por que isso importa) <<<

A estratégia de recuperação é INJETADA (`StrategyFn`), não chamada direto. Com
isso o runner não sabe — nem se importa — se os chunks vieram do pgvector real ou
de um fake de teste. Resultado: dá para testar a orquestração e a agregação SEM
banco e SEM LLM (injetando estratégia-fake + mock das métricas), e o mesmo runner
roda em produção apontando para o banco de verdade. É o mesmo princípio
"primitiva agnóstica à origem" que guiou search_by_embedding/rerank/generate.

Camadas, deliberadamente separadas:
    run_evaluation -> dado CRU por sample (list[SampleResult]) — preserva o caso
                      individual; uma média esconde QUAL pergunta regrediu.
    aggregate      -> média por (estratégia, métrica) — função PURA.
    format_report  -> tabela markdown — função PURA; re-formata sem re-rodar nada.
"""

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from finsight.evals.dataset import SEED_DATASET, EvalDataset, EvalSample
from finsight.evals.generator import generate_answer
from finsight.evals.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from finsight.retrieval.hyde import hyde_retrieve
from finsight.retrieval.reranker import retrieve_and_rerank
from finsight.retrieval.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

# Ordem canônica das métricas nas colunas do relatório. Fixa para a tabela ser
# estável entre execuções (e os testes poderem casar string exata).
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# Estratégia de recuperação como função injetável
# ---------------------------------------------------------------------------


class StrategyFn(Protocol):
    """
    Assinatura ÚNICA de uma estratégia de recuperação.

    `retrieve`, `hyde_retrieve` e `retrieve_and_rerank` têm assinaturas LEVEMENTE
    diferentes (top_k vs fetch_k/top_n, use_hyde...). Em vez de o runner conhecer
    cada uma, definimos UM contrato — `(query, *, ticker, session) -> chunks` — e
    adaptamos cada técnica a ele com um wrapper fino (abaixo). O runner só fala
    este Protocol; trocar/adicionar estratégia não toca no laço de avaliação.
    """

    def __call__(
        self,
        query: str,
        *,
        ticker: str | None,
        session: AsyncSession | None,
    ) -> Awaitable[list[RetrievedChunk]]: ...


async def _strategy_baseline(
    query: str, *, ticker: str | None, session: AsyncSession | None
) -> list[RetrievedChunk]:
    """Busca vetorial crua — o piso de comparação (sem HyDE, sem re-rank)."""
    return await retrieve(query, ticker=ticker, session=session)


async def _strategy_hyde(
    query: str, *, ticker: str | None, session: AsyncSession | None
) -> list[RetrievedChunk]:
    """HyDE: embeda um documento hipotético em vez da query crua (recall esperto)."""
    return await hyde_retrieve(query, ticker=ticker, session=session)


async def _strategy_rerank(
    query: str, *, ticker: str | None, session: AsyncSession | None
) -> list[RetrievedChunk]:
    """Over-fetch vetorial -> re-rank LLM listwise (ordenação fina sobre a busca crua)."""
    return await retrieve_and_rerank(query, ticker=ticker, use_hyde=False, session=session)


# As três estratégias da comparação central da semana. É um dict (nome -> fn) de
# propósito: o runner itera sobre ele, então adicionar "hyde+rerank" no futuro é
# uma linha aqui, zero mudança no laço.
DEFAULT_STRATEGIES: dict[str, StrategyFn] = {
    "baseline": _strategy_baseline,
    "hyde": _strategy_hyde,
    "rerank": _strategy_rerank,
}


# ---------------------------------------------------------------------------
# Resultado por sample
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """
    A avaliação de UM sample sob UMA estratégia.

    Guardamos `answer` junto dos scores porque, ao investigar uma nota baixa, a
    primeira pergunta é "o que o sistema respondeu?". `scores` mapeia nome da
    métrica -> [0, 1]. Mantemos o grão fino (sample x estratégia) e só agregamos
    depois — assim dá para ver QUAL pergunta puxou a média para baixo.
    """

    sample_id: str
    strategy: str
    scores: dict[str, float]
    answer: str = ""


# ---------------------------------------------------------------------------
# Núcleo: avaliar um sample sob uma estratégia
# ---------------------------------------------------------------------------


async def evaluate_sample(
    sample: EvalSample,
    strategy: StrategyFn,
    *,
    strategy_name: str,
    session: AsyncSession | None = None,
) -> SampleResult:
    """
    Roda o pipeline completo para um sample: recupera -> gera -> mede as 4 métricas.

    Os argumentos de cada métrica espelham EXATAMENTE o mapa runtime-gabarito que
    fixamos em dataset.py — é aqui que o contrato do golden set "ganha vida":
        faithfulness     <- (answer, contexts)                    [auto-contida]
        answer_relevancy <- (question, answer)                    [auto-contida]
        context_precision<- (question, contexts, gt_contexts)     [usa gabarito]
        context_recall   <- (contexts, ground_truth)             [usa gabarito]

    As 4 métricas são independentes -> rodamos com asyncio.gather (4 chamadas de
    LLM/embeddings em paralelo). A geração tem de vir ANTES (faithfulness e
    answer_relevancy dependem da resposta), então gather só envolve as métricas.
    """
    chunks = await strategy(sample.question, ticker=sample.ticker, session=session)
    contexts = [c.content for c in chunks]

    # O "G": resposta ancorada nos chunks que ESTA estratégia trouxe.
    answer = await generate_answer(sample.question, contexts)

    # As 4 réguas em paralelo — nenhuma depende do resultado da outra.
    faith, ans_rel, ctx_prec, ctx_rec = await asyncio.gather(
        faithfulness(answer, contexts),
        answer_relevancy(sample.question, answer),
        context_precision(sample.question, contexts, sample.ground_truth_contexts),
        context_recall(contexts, sample.ground_truth),
    )

    scores = {
        "faithfulness": faith.score,
        "answer_relevancy": ans_rel.score,
        "context_precision": ctx_prec.score,
        "context_recall": ctx_rec.score,
    }
    logger.debug("Avaliado %s sob %s: %s", sample.id, strategy_name, scores)
    return SampleResult(
        sample_id=sample.id,
        strategy=strategy_name,
        scores=scores,
        answer=answer,
    )


async def run_evaluation(
    dataset: EvalDataset,
    strategies: dict[str, StrategyFn] | None = None,
    *,
    session: AsyncSession | None = None,
) -> list[SampleResult]:
    """
    Avalia TODO o dataset sob TODAS as estratégias. Devolve o dado cru por sample.

    Itera estratégia x sample sequencialmente. Poderíamos paralelizar com gather,
    mas mantemos sequencial DE PROPÓSITO: o custo aqui é dominado por chamadas de
    LLM (re-rank, geração, 4 métricas), e disparar tudo de uma vez estouraria
    rate limit da OpenAI com um golden set realista. Sequencial é previsível e
    suave no orçamento — e a avaliação é offline, latência não importa.

    Não captura exceções de sample: se uma estratégia quebrar, queremos saber
    (avaliação é diagnóstico — erro silencioso aqui mascara regressão real).
    """
    strategies = strategies if strategies is not None else DEFAULT_STRATEGIES
    results: list[SampleResult] = []

    for strategy_name, strategy_fn in strategies.items():
        logger.info("Avaliando estratégia %r sobre %d samples", strategy_name, len(dataset))
        for sample in dataset.samples:
            result = await evaluate_sample(
                sample, strategy_fn, strategy_name=strategy_name, session=session
            )
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Agregação e relatório — funções PURAS (sem mock para testar)
# ---------------------------------------------------------------------------


@dataclass
class StrategyReport:
    """Médias de uma estratégia: métrica -> média no dataset, + nº de samples."""

    strategy: str
    means: dict[str, float]
    n_samples: int = 0


def aggregate(results: list[SampleResult]) -> dict[str, StrategyReport]:
    """
    Reduz os resultados por-sample a médias por (estratégia, métrica).

    Função pura: entra lista de SampleResult, sai um relatório por estratégia.
    Agrupa por estratégia preservando a ordem de primeira aparição (dict mantém
    ordem de inserção) — assim a tabela sai na ordem em que as estratégias rodaram.
    Estratégia sem samples não entra (evita divisão por zero).
    """
    # estratégia -> métrica -> lista de scores observados
    buckets: dict[str, dict[str, list[float]]] = {}
    for r in results:
        per_metric = buckets.setdefault(r.strategy, {m: [] for m in METRIC_NAMES})
        for metric, score in r.scores.items():
            per_metric.setdefault(metric, []).append(score)

    reports: dict[str, StrategyReport] = {}
    for strategy, per_metric in buckets.items():
        # n de samples = quantos scores caíram em qualquer métrica (todas têm o
        # mesmo n, pois cada sample contribui com as 4). Usamos a 1ª como amostra.
        first = next((v for v in per_metric.values() if v), [])
        means = {
            metric: (sum(scores) / len(scores) if scores else 0.0)
            for metric, scores in per_metric.items()
        }
        reports[strategy] = StrategyReport(strategy=strategy, means=means, n_samples=len(first))
    return reports


def format_report(results: list[SampleResult]) -> str:
    """
    Monta a tabela markdown comparativa — uma linha por estratégia, uma coluna por
    métrica. É o ENTREGÁVEL visual da semana: lê-se na diagonal se HyDE/rerank
    ganharam de baseline, e em qual métrica.

    Pura: deriva tudo de `aggregate(results)`. Não imprime nem grava — devolve a
    string; quem chama decide (print, arquivo, log).
    """
    reports = aggregate(results)
    header = "| strategy | n | " + " | ".join(METRIC_NAMES) + " |"
    sep = "|" + "---|" * (len(METRIC_NAMES) + 2)
    lines = [header, sep]
    for report in reports.values():
        cells = " | ".join(f"{report.means.get(m, 0.0):.3f}" for m in METRIC_NAMES)
        lines.append(f"| {report.strategy} | {report.n_samples} | {cells} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrada manual (precisa de DB + OpenAI reais)
# ---------------------------------------------------------------------------


async def run_default() -> str:
    """
    Roda a comparação padrão sobre o SEED_DATASET e devolve a tabela.

    ATENÇÃO: bate no pgvector real e na OpenAI real — NÃO roda no CI. É o atalho
    para você rodar a suíte localmente (`python -m finsight.evals.runner`) depois
    de subir o Postgres (porta 5433) e ingerir os PDFs. O SEED é fictício
    ("Petro Norte"); troque pelo golden set real para números que signifiquem algo.
    """
    results = await run_evaluation(SEED_DATASET)
    return format_report(results)


if __name__ == "__main__":  # pragma: no cover
    print(asyncio.run(run_default()))
