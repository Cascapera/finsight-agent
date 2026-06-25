"""
Testes do RAG Agent (Semana 6, Passo 1) — sem rede/banco.

O RAG Agent é uma casca de nó sobre a camada de retrieval (Semana 3). Mockamos o
ponto de integração `retrieve_and_rerank` (o único que toca banco/LLM) e checamos
o contrato de nó: patch {"rag": RAGOutput} no sucesso, vazio sem erro, e captura de
erro do retrieval em state["errors"] (o nó nunca propaga).
"""

import uuid
from typing import Any

import pytest

from finsight.agents import rag
from finsight.agents.rag import rag_node
from finsight.graph.state import RAGOutput
from finsight.retrieval.retriever import RetrievedChunk


def _make_state(ticker: str = "PETR4", query: str = "qual o endividamento?") -> Any:
    """AgentState mínimo: o rag_node lê `ticker` e `query`."""
    return {"ticker": ticker, "query": query, "asset_type": "stock"}


def _chunks(n: int = 2) -> list[RetrievedChunk]:
    """Chunks recuperados fabricados, com score decrescente (como o retrieval entrega)."""
    return [
        RetrievedChunk(
            content=f"trecho {i}",
            score=1.0 - i * 0.1,
            document_id=uuid.uuid4(),
            document_title=f"Relatorio {i}",
            chunk_index=i,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_node_returns_rag_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sucesso: o nó converte os chunks em RAGOutput (listas paralelas) e não erra."""
    chunks = _chunks(2)

    async def fake_retrieve(query: str, *, ticker: str, use_hyde: bool) -> list[RetrievedChunk]:
        assert ticker == "PETR4"
        return chunks

    monkeypatch.setattr(rag, "retrieve_and_rerank", fake_retrieve)

    patch = await rag_node(_make_state("PETR4"))

    assert "errors" not in patch
    out = patch["rag"]
    assert isinstance(out, RAGOutput)
    # Listas paralelas por índice: chunk/source/score descrevem o mesmo trecho.
    assert out.chunks == ["trecho 0", "trecho 1"]
    assert out.sources == ["Relatorio 0", "Relatorio 1"]
    assert out.relevance_scores == [pytest.approx(1.0), pytest.approx(0.9)]


@pytest.mark.asyncio
async def test_node_empty_result_no_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem trechos: RAGOutput vazio, SEM erro (não há nada para o ativo, não é falha)."""

    async def fake_retrieve(query: str, *, ticker: str, use_hyde: bool) -> list[RetrievedChunk]:
        return []

    monkeypatch.setattr(rag, "retrieve_and_rerank", fake_retrieve)

    patch = await rag_node(_make_state())

    assert "errors" not in patch
    assert patch["rag"].chunks == []


@pytest.mark.asyncio
async def test_node_captures_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falha no retrieval (banco/LLM): o nó captura — RAGOutput vazio + erro registrado."""

    async def boom(query: str, *, ticker: str, use_hyde: bool) -> list[RetrievedChunk]:
        raise RuntimeError("pgvector offline")

    monkeypatch.setattr(rag, "retrieve_and_rerank", boom)

    patch = await rag_node(_make_state("VALE3"))

    assert isinstance(patch["rag"], RAGOutput)
    assert patch["rag"].chunks == []
    assert len(patch["errors"]) == 1
    assert "VALE3" in patch["errors"][0]
