"""
Testes do Risk Agent (Semana 5, Passo 3) — sem rede.

  * synthesize: o núcleo de síntese. Mockamos `_get_chat_client` (mesmo truque dos
    outros nós LLM) e checamos: a análise do veredito sai no FinalAnswer, o
    disclaimer vem do CÓDIGO (não do LLM), e dados parciais (financial None) não
    quebram a serialização.
  * risk_node: o contrato de nó — patch {"final_answer": ...} no sucesso e captura
    de erro do LLM em state["errors"] (nunca propaga).
"""

from typing import Any

import pytest

from finsight.agents import risk
from finsight.agents.risk import _RiskVerdict, risk_node, synthesize
from finsight.graph.state import FinalAnswer, FinancialOutput, ResearchOutput


def _make_state(**overrides: Any) -> Any:
    """AgentState mínimo para a síntese, com outputs preenchidos por padrão."""
    state: dict[str, Any] = {
        "query": "vale a pena?",
        "ticker": "PETR4",
        "asset_type": "stock",
        "financial": FinancialOutput(current_price=30.0, sharpe_ratio=1.2, var_95=0.03),
        "research": ResearchOutput(
            summary="resultados sólidos",
            sentiment="bullish",
            key_events=["lucro recorde"],
            confidence=0.8,
            sources=["https://news.example/0"],
        ),
        "rag": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Fakes do LLM — runnable determinístico, padrão dos outros testes de nó LLM
# ---------------------------------------------------------------------------


class _FixedStructured:
    def __init__(self, verdict: _RiskVerdict) -> None:
        self._verdict = verdict

    async def ainvoke(self, _messages: Any) -> _RiskVerdict:
        return self._verdict


class _FakeChatClient:
    def __init__(self, verdict: _RiskVerdict) -> None:
        self._verdict = verdict

    def with_structured_output(self, _schema: Any) -> _FixedStructured:
        return _FixedStructured(self._verdict)


class _ExplodingChatClient:
    """Faz o synthesize estourar — simula o LLM fora do ar para o teste do nó."""

    def with_structured_output(self, _schema: Any) -> Any:
        raise RuntimeError("openai 503")


# ===========================================================================
# synthesize — núcleo
# ===========================================================================


@pytest.mark.asyncio
async def test_synthesize_builds_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """O veredito do LLM vira FinalAnswer; disclaimer NÃO vem do modelo."""
    verdict = _RiskVerdict(
        analysis="Sharpe alto e sentimento bullish sugerem oportunidade, com VaR moderado.",
        key_points=["Sharpe 1.2", "sentimento bullish"],
        risk_factors=["VaR 3% diário"],
    )
    monkeypatch.setattr(risk, "_get_chat_client", lambda: _FakeChatClient(verdict))

    out = await synthesize(_make_state())

    assert isinstance(out, FinalAnswer)
    assert out.analysis == verdict.analysis
    assert out.key_points == verdict.key_points
    assert out.risk_factors == verdict.risk_factors
    # disclaimer é o default fixo do schema — garantido pelo código, não pelo LLM.
    assert "não constitui recomendação" in out.disclaimer.lower()


@pytest.mark.asyncio
async def test_synthesize_handles_partial_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com financial=None e rag=None a serialização não quebra — síntese roda igual."""
    verdict = _RiskVerdict(
        analysis="Dados quantitativos indisponíveis; análise baseada apenas em notícias.",
        key_points=["sentimento bullish"],
        risk_factors=["sem métricas de risco quantitativas"],
    )
    monkeypatch.setattr(risk, "_get_chat_client", lambda: _FakeChatClient(verdict))

    out = await synthesize(_make_state(financial=None))

    assert isinstance(out, FinalAnswer)
    assert out.analysis == verdict.analysis


# ===========================================================================
# risk_node — contrato de nó
# ===========================================================================


@pytest.mark.asyncio
async def test_node_returns_final_answer_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sucesso: o nó devolve {"final_answer": FinalAnswer} e nenhum erro."""
    verdict = _RiskVerdict(analysis="ok", key_points=["p"], risk_factors=["r"])
    monkeypatch.setattr(risk, "_get_chat_client", lambda: _FakeChatClient(verdict))

    patch = await risk_node(_make_state())

    assert "errors" not in patch
    assert isinstance(patch["final_answer"], FinalAnswer)
    assert patch["final_answer"].analysis == "ok"


@pytest.mark.asyncio
async def test_node_captures_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falha no LLM da síntese: fallback + erro registrado, nunca propaga."""
    monkeypatch.setattr(risk, "_get_chat_client", lambda: _ExplodingChatClient())

    patch = await risk_node(_make_state(ticker="VALE3"))

    assert isinstance(patch["final_answer"], FinalAnswer)
    assert patch["final_answer"].key_points == []
    assert len(patch["errors"]) == 1
    assert "VALE3" in patch["errors"][0]
    # mesmo no fallback, o disclaimer obrigatório está presente.
    assert patch["final_answer"].disclaimer
