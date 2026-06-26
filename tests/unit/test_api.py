"""
Testes da API (Semana 6, Passo 3) — endpoint SSE sem rede.

Sobe a app com o TestClient do FastAPI e mocka os pontos de rede dos nós (resolvidos
em tempo de chamada -> monkeypatch pega). Verifica o /health e o stream do /analyze:
as linhas `event: progress` / `event: complete` e o final_answer no corpo SSE.
"""

import uuid
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finsight.agents import financial, rag, research, risk
from finsight.agents.research import NewsItem, _SentimentVerdict
from finsight.agents.risk import _RiskVerdict
from finsight.api.app import create_app
from finsight.retrieval.retriever import RetrievedChunk


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
    """Mocka os pontos de rede dos 3 ramos + síntese (mesmo padrão do test_orchestrator)."""
    prices = pd.Series([100.0 * (1.0 + 0.001 * (i % 5 - 2)) ** i for i in range(60)])
    monkeypatch.setattr(financial, "_fetch_prices", lambda _ticker: prices)

    monkeypatch.setattr(
        research,
        "_search_news",
        lambda _t, _q: [NewsItem(title="t", url="https://news.example/0", content="c")],
    )
    research_verdict = _SentimentVerdict(
        summary="bom momento", sentiment="bullish", key_events=["lucro"], confidence=0.8
    )
    monkeypatch.setattr(research, "_get_chat_client", lambda: _FakeChatClient(research_verdict))

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

    risk_verdict = _RiskVerdict(
        analysis="síntese integrada", key_points=["p1"], risk_factors=["r1"]
    )
    monkeypatch.setattr(risk, "_get_chat_client", lambda: _FakeChatClient(risk_verdict))


def test_health() -> None:
    """O liveness probe responde 200 com status ok."""
    with TestClient(create_app()) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_rejects_empty_query() -> None:
    """Validação na borda: query vazia -> 422 (Pydantic), sem tocar o grafo."""
    with TestClient(create_app()) as client:
        resp = client.post("/analyze", json={"query": "   ", "ticker": "PETR4"})
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_prometheus() -> None:
    """GET /metrics responde no formato Prometheus com os nomes das nossas métricas."""
    with TestClient(create_app()) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "finsight_node_duration_seconds" in resp.text


def test_metrics_endpoint_404_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com prometheus_enabled=False, /metrics responde 404 (operador optou por não expor)."""
    from finsight.api import routes

    monkeypatch.setattr(routes.settings, "prometheus_enabled", False)
    with TestClient(create_app()) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 404


def test_analyze_streams_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /analyze faz streaming dos eventos SSE até o complete com final_answer."""
    _wire_happy_path(monkeypatch)

    with TestClient(create_app()) as client:
        resp = client.post("/analyze", json={"query": "vale a pena?", "ticker": "petr4"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    body = resp.text
    # Os tipos de evento aparecem como nome do evento SSE.
    assert "event: progress" in body
    assert "event: complete" in body
    # O final_answer chega no payload (ticker foi normalizado para maiúsculas).
    assert "síntese integrada" in body
