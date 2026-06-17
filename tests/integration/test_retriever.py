"""
Teste de integração do retriever: busca por similaridade no pgvector.

Requer Postgres com pgvector rodando e migrado:
    docker compose up -d postgres
    alembic upgrade head

Roda com:
    pytest -m integration

Diferença-chave para o teste de ingestão: aqui os embeddings precisam ser
DISTINTOS entre si, senão todas as distâncias seriam iguais e a ordenação por
similaridade não teria o que provar. Por isso o fake mapeia palavra-chave ->
dimensão: cada conteúdo vira um vetor unitário numa direção própria, e a query
"aponta" para a direção do chunk que deve vencer.

O retriever recebe a `session` do fixture (rollback) via injeção — assim lemos
dados só flushados, sem commit (ver retriever._session_scope).
"""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from finsight.db.session import AsyncSessionLocal
from finsight.ingestion.chunker import ChunkResult
from finsight.ingestion.embedder import EMBEDDING_DIM, embed_chunks
from finsight.ingestion.indexer import index_document
from finsight.retrieval.retriever import RetrievedChunk, retrieve, to_rag_output

# ---------------------------------------------------------------------------
# Fake de embeddings ciente do conteúdo
# ---------------------------------------------------------------------------

# Cada palavra-chave reserva UMA dimensão. Vetores unitários em dimensões
# diferentes são ORTOGONAIS (similaridade de cosseno = 0); o mesmo unitário
# contra si mesmo dá similaridade 1. Isso nos dá controle total sobre quem é
# "mais parecido" com quem, sem depender de um modelo de embedding real.
_KEYWORD_DIM = {"lucro": 0, "dívida": 1, "caixa": 2}


def _unit(index: int, dim: int = EMBEDDING_DIM) -> list[float]:
    """Vetor unitário: 1.0 na dimensão `index`, 0.0 no resto."""
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


def _embedding_for(text: str) -> list[float]:
    """Mapeia o texto para um vetor pela primeira palavra-chave encontrada."""
    lowered = text.lower()
    for keyword, index in _KEYWORD_DIM.items():
        if keyword in lowered:
            return _unit(index)
    # Bucket neutro para textos sem palavra-chave — ortogonal a todos acima.
    return _unit(3)


def _make_fake_create() -> Any:
    """Substituto async de client.embeddings.create, determinístico por conteúdo."""

    async def fake_create(*, model: str, input: list[str]) -> Any:
        data = [SimpleNamespace(index=i, embedding=_embedding_for(t)) for i, t in enumerate(input)]
        usage = SimpleNamespace(total_tokens=sum(len(t.split()) for t in input))
        return SimpleNamespace(data=data, usage=usage)

    return fake_create


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    """Sessão async que reverte tudo no fim — isolamento sem TRUNCATE."""
    async with AsyncSessionLocal() as s:
        try:
            yield s
        finally:
            await s.rollback()


async def _seed_document(
    session: AsyncSession,
    *,
    ticker: str,
    title: str,
    contents: list[str],
) -> Any:
    """Indexa um documento com os `contents` dados (já embedados) e flusha."""
    chunks = [
        ChunkResult(content=text, chunk_index=i, metadata={"ticker": ticker})
        for i, text in enumerate(contents)
    ]
    embedded = await embed_chunks(chunks)
    document = await index_document(
        session, ticker=ticker, title=title, chunks=embedded, source_type="earnings"
    )
    await session.flush()
    return document


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_retrieve_orders_by_similarity(session: AsyncSession, mocker: Any) -> None:
    """O chunk cuja direção bate com a da query vem primeiro, score ~1.0."""
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=_make_fake_create()))
    mocker.patch("finsight.ingestion.embedder._get_client", return_value=fake_client)

    await _seed_document(
        session,
        ticker="PETR4",
        title="Relatório PETR4 Q3",
        contents=[
            "O lucro líquido cresceu 20% no trimestre.",  # dim 0
            "A dívida bruta aumentou frente ao ano anterior.",  # dim 1
            "O fluxo de caixa livre ficou positivo.",  # dim 2
        ],
    )

    results = await retrieve(
        "Qual foi o lucro líquido reportado?",  # casa com a dimensão 0
        ticker="PETR4",
        top_k=3,
        session=session,
    )

    assert len(results) == 3
    # O mais relevante é o chunk do "lucro" — mesma direção da query.
    assert "lucro" in results[0].content.lower()
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    # Scores em ordem decrescente (invariante central do retriever).
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # Os ortogonais ficam perto de 0.
    assert results[-1].score == pytest.approx(0.0, abs=1e-4)


@pytest.mark.integration
async def test_retrieve_filters_by_ticker(session: AsyncSession, mocker: Any) -> None:
    """Com ticker informado, chunks de outros ativos não aparecem."""
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=_make_fake_create()))
    mocker.patch("finsight.ingestion.embedder._get_client", return_value=fake_client)

    petr = await _seed_document(
        session,
        ticker="PETR4",
        title="Relatório PETR4",
        contents=["O lucro líquido da Petrobras cresceu."],
    )
    await _seed_document(
        session,
        ticker="VALE3",
        title="Relatório VALE3",
        contents=["O lucro líquido da Vale cresceu."],
    )

    results = await retrieve("lucro líquido", ticker="PETR4", top_k=10, session=session)

    assert len(results) == 1
    assert results[0].document_id == petr.id
    assert results[0].document_title == "Relatório PETR4"


@pytest.mark.integration
async def test_retrieve_empty_query_returns_empty(session: AsyncSession) -> None:
    """Query em branco curto-circuita: sem embedding, sem ida ao banco."""
    assert await retrieve("   ", session=session) == []


def test_to_rag_output_maps_parallel_lists() -> None:
    """to_rag_output achata os objetos ricos em três listas paralelas por índice."""
    import uuid

    chunks = [
        RetrievedChunk(
            content="trecho A",
            score=0.9,
            document_id=uuid.uuid4(),
            document_title="Doc A",
            chunk_index=0,
        ),
        RetrievedChunk(
            content="trecho B",
            score=0.5,
            document_id=uuid.uuid4(),
            document_title="Doc B",
            chunk_index=1,
        ),
    ]

    out = to_rag_output(chunks)

    assert out.chunks == ["trecho A", "trecho B"]
    assert out.sources == ["Doc A", "Doc B"]
    assert out.relevance_scores == [0.9, 0.5]
