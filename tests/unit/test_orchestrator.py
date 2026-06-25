"""
Testes do Orchestrator (Semana 5, Passo 3) — grafo end-to-end SEM rede.

A graça aqui é provar a ORQUESTRAÇÃO: o diamante roda os dois ramos e converge na
síntese, e os erros dos ramos paralelos se acumulam. Para isso mockamos os pontos
de rede de CADA nó (resolvidos em tempo de chamada, então o monkeypatch pega):
  - financial._fetch_prices  (yfinance)
  - research._search_news    (Tavily)
  - research._get_chat_client e risk._get_chat_client (OpenAI)

`build_initial_state` é testado puro (sem grafo).
"""

import uuid
from typing import Any

import pandas as pd
import pytest

from finsight.agents import financial, rag, research, risk
from finsight.agents.research import NewsItem, _SentimentVerdict
from finsight.agents.risk import _RiskVerdict
from finsight.graph.orchestrator import build_initial_state, run_analysis
from finsight.retrieval.retriever import RetrievedChunk

# ---------------------------------------------------------------------------
# Fakes de LLM compartilhados (cada client devolve o veredito do seu schema)
# ---------------------------------------------------------------------------


class _FixedStructured:
    def __init__(self, verdict: Any) -> None:
        self._verdict = verdict

    async def ainvoke(self, _messages: Any) -> Any:
        return self._verdict


class _FakeChatClient:
    def __init__(self, verdict: Any) -> None:
        self._verdict = verdict

    def with_structured_output(self, _schema: Any) -> _FixedStructured:
        return _FixedStructured(self._verdict)


def _wire_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocka os pontos de rede dos 3 ramos + síntese: caminho feliz, sem rede."""
    # Financial: série de preços com variação real -> métricas calculáveis.
    prices = pd.Series([100.0 * (1.0 + 0.001 * (i % 5 - 2)) ** i for i in range(60)])
    monkeypatch.setattr(financial, "_fetch_prices", lambda _ticker: prices)

    # Research: notícias fabricadas + veredito de sentimento fixo.
    monkeypatch.setattr(
        research,
        "_search_news",
        lambda _t, _q: [NewsItem(title="t", url="https://news.example/0", content="c")],
    )
    research_verdict = _SentimentVerdict(
        summary="bom momento", sentiment="bullish", key_events=["lucro"], confidence=0.8
    )
    monkeypatch.setattr(research, "_get_chat_client", lambda: _FakeChatClient(research_verdict))

    # RAG: chunks recuperados fabricados (mocka a camada de retrieval inteira).
    async def fake_retrieve(query: str, *, ticker: str, use_hyde: bool) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                content="trecho relevante",
                score=0.9,
                document_id=uuid.uuid4(),
                document_title="Relatorio Anual",
                chunk_index=0,
            )
        ]

    monkeypatch.setattr(rag, "retrieve_and_rerank", fake_retrieve)

    # Risk: veredito de síntese fixo.
    risk_verdict = _RiskVerdict(
        analysis="síntese integrada", key_points=["p1"], risk_factors=["r1"]
    )
    monkeypatch.setattr(risk, "_get_chat_client", lambda: _FakeChatClient(risk_verdict))


# ===========================================================================
# build_initial_state — puro
# ===========================================================================


def test_build_initial_state_shape() -> None:
    """Inputs preenchidos, outputs None, errors lista vazia, execution_id presente."""
    state = build_initial_state("vale a pena?", "PETR4", "stock")

    assert state["query"] == "vale a pena?"
    assert state["ticker"] == "PETR4"
    assert state["asset_type"] == "stock"
    assert state["financial"] is None
    assert state["research"] is None
    assert state["final_answer"] is None
    assert state["cached"] is False
    assert state["errors"] == []
    assert state["execution_id"]  # uuid4 não-vazio


# ===========================================================================
# Grafo end-to-end
# ===========================================================================


@pytest.mark.asyncio
async def test_graph_runs_fanout_to_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caminho feliz: os três ramos rodam e a síntese converge num final_answer."""
    _wire_happy_path(monkeypatch)

    result = await run_analysis("vale a pena?", "PETR4", "stock")

    # Fan-out preencheu os três ramos...
    assert result["financial"] is not None
    assert result["financial"].current_price is not None
    assert result["research"] is not None
    assert result["research"].sentiment == "bullish"
    assert result["rag"] is not None
    assert result["rag"].chunks == ["trecho relevante"]
    # ...e o fan-in sintetizou.
    assert result["final_answer"] is not None
    assert result["final_answer"].analysis == "síntese integrada"
    assert result["final_answer"].disclaimer  # disclaimer fixo presente
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_graph_accumulates_branch_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Um ramo falha (Tavily cai), o outro segue: o erro é ACUMULADO via reducer `add`
    e a síntese AINDA roda (degradação graciosa de ponta a ponta).
    """
    _wire_happy_path(monkeypatch)

    # Sobrescreve só o ramo de research para falhar.
    def boom(_t: str, _q: str) -> list[NewsItem]:
        raise RuntimeError("tavily offline")

    monkeypatch.setattr(research, "_search_news", boom)

    result = await run_analysis("e aí?", "VALE3", "stock")

    # Financial seguiu normalmente.
    assert result["financial"] is not None
    assert result["financial"].current_price is not None
    # Research degradou para neutro, mas o erro foi registrado.
    assert result["research"].sentiment == "neutral"
    assert any("VALE3" in e for e in result["errors"])
    # A síntese rodou apesar da falha parcial.
    assert result["final_answer"] is not None
