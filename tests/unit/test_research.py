"""
Testes do Research Agent (Semana 5, Passo 2) — sem rede.

Duas frentes, espelhando a separação do módulo:
  * analyze_sentiment: o núcleo de raciocínio. Mockamos `_get_chat_client` com um
    fake cujo `.with_structured_output(schema)` devolve um runnable determinístico
    (mesmo truque do test_metrics) e checamos a classificação + a colagem das
    fontes REAIS (não do LLM) + o guard de notícias-vazias (não chama o LLM).
  * research_node: o contrato de nó. Mockamos `_search_news` e `analyze_sentiment`
    para isolar a lógica do nó — patch no sucesso e captura de erro de CADA ponto
    de rede em state["errors"] (o nó nunca propaga).
"""

from typing import Any

import pytest

from finsight.agents import research
from finsight.agents.research import (
    NewsItem,
    _SentimentVerdict,
    analyze_sentiment,
    research_node,
)
from finsight.graph.state import ResearchOutput


def _make_state(ticker: str = "PETR4", query: str = "como está o ativo") -> Any:
    """AgentState mínimo: o research_node lê `ticker` e `query`."""
    return {"ticker": ticker, "query": query, "asset_type": "stock"}


def _news(n: int = 2) -> list[NewsItem]:
    """Notícias fabricadas com URLs conhecidas, para checar a colagem de fontes."""
    return [
        NewsItem(title=f"Notícia {i}", url=f"https://news.example/{i}", content=f"corpo {i}")
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fakes do LLM — runnable determinístico, no padrão do test_metrics
# ---------------------------------------------------------------------------


class _FixedStructured:
    """Runnable falso devolvido por `.with_structured_output`: sempre o mesmo veredito."""

    def __init__(self, verdict: _SentimentVerdict) -> None:
        self._verdict = verdict

    async def ainvoke(self, _messages: Any) -> _SentimentVerdict:
        return self._verdict


class _FakeChatClient:
    """Client cru falso: `.with_structured_output(schema)` -> runnable determinístico."""

    def __init__(self, verdict: _SentimentVerdict) -> None:
        self._verdict = verdict

    def with_structured_output(self, _schema: Any) -> _FixedStructured:
        return _FixedStructured(self._verdict)


class _ExplodingChatClient:
    """Prova que um guard NÃO chamou o LLM: qualquer uso estoura."""

    def with_structured_output(self, _schema: Any) -> Any:
        raise AssertionError("LLM não deveria ser chamado neste caminho")


# ===========================================================================
# analyze_sentiment — núcleo de raciocínio
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("sentiment", ["bullish", "bearish", "neutral"])
async def test_analyze_classifies_sentiment(
    monkeypatch: pytest.MonkeyPatch, sentiment: str
) -> None:
    """O sentimento do veredito do LLM aparece no ResearchOutput, para os 3 valores."""
    verdict = _SentimentVerdict(
        summary="resumo",
        sentiment=sentiment,  # type: ignore[arg-type]
        key_events=["evento A"],
        confidence=0.8,
    )
    monkeypatch.setattr(research, "_get_chat_client", lambda: _FakeChatClient(verdict))

    out = await analyze_sentiment("e aí?", _news(2))

    assert isinstance(out, ResearchOutput)
    assert out.sentiment == sentiment
    assert out.confidence == pytest.approx(0.8)
    assert out.key_events == ["evento A"]


@pytest.mark.asyncio
async def test_sources_come_from_news_not_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Decisão-chave: as `sources` são as URLs REAIS das notícias, nunca do LLM.
    O veredito não tem campo de fontes; mesmo assim o output traz as URLs do Tavily.
    """
    verdict = _SentimentVerdict(
        summary="resumo", sentiment="bullish", key_events=[], confidence=0.5
    )
    monkeypatch.setattr(research, "_get_chat_client", lambda: _FakeChatClient(verdict))

    news = _news(3)
    out = await analyze_sentiment("q", news)

    assert out.sources == [item.url for item in news]


@pytest.mark.asyncio
async def test_no_news_short_circuits_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem notícias: output neutro determinístico, SEM tocar o LLM."""
    monkeypatch.setattr(research, "_get_chat_client", lambda: _ExplodingChatClient())

    out = await analyze_sentiment("q", [])

    assert out.sentiment == "neutral"
    assert out.confidence == pytest.approx(0.0)
    assert out.sources == []
    assert out.key_events == []


# ===========================================================================
# research_node — contrato de nó
# ===========================================================================


@pytest.mark.asyncio
async def test_node_returns_research_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sucesso: o nó devolve {"research": ResearchOutput} e nenhum erro."""
    expected = ResearchOutput(
        summary="ok",
        sentiment="bullish",
        key_events=["resultado trimestral"],
        confidence=0.9,
        sources=["https://news.example/0"],
    )

    def fake_search(ticker: str, query: str) -> list[NewsItem]:
        assert ticker == "PETR4"
        return _news(1)

    async def fake_analyze(query: str, news: list[NewsItem]) -> ResearchOutput:
        return expected

    monkeypatch.setattr(research, "_search_news", fake_search)
    monkeypatch.setattr(research, "analyze_sentiment", fake_analyze)

    patch = await research_node(_make_state("PETR4"))

    assert "errors" not in patch
    assert patch["research"] is expected


@pytest.mark.asyncio
async def test_node_captures_search_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falha no Tavily: o nó NÃO propaga — captura e escreve em errors."""

    def boom(ticker: str, query: str) -> list[NewsItem]:
        raise RuntimeError("tavily offline")

    monkeypatch.setattr(research, "_search_news", boom)

    patch = await research_node(_make_state("VALE3"))

    assert isinstance(patch["research"], ResearchOutput)
    assert patch["research"].sentiment == "neutral"
    assert len(patch["errors"]) == 1
    assert "VALE3" in patch["errors"][0]


@pytest.mark.asyncio
async def test_node_captures_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falha no LLM (após a busca): também vira erro registrado, output neutro."""

    def fake_search(ticker: str, query: str) -> list[NewsItem]:
        return _news(2)

    async def boom(query: str, news: list[NewsItem]) -> ResearchOutput:
        raise RuntimeError("openai 503")

    monkeypatch.setattr(research, "_search_news", fake_search)
    monkeypatch.setattr(research, "analyze_sentiment", boom)

    patch = await research_node(_make_state("ITUB4"))

    assert patch["research"].sentiment == "neutral"
    assert patch["research"].confidence == pytest.approx(0.0)
    assert "ITUB4" in patch["errors"][0]
