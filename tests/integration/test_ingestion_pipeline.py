"""
Teste de integração do pipeline de ingestão: embed_chunks -> index_document.

Requer Postgres com a extensão pgvector rodando e migrado:
    docker compose up -d postgres
    alembic upgrade head

Roda com:
    pytest -m integration

O client da OpenAI é mockado (CI não faz chamadas reais), mas o banco é real —
validamos o INSERT na coluna Vector(1536) e a leitura de volta.
"""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finsight.db.models import Document, DocumentChunk
from finsight.db.session import AsyncSessionLocal
from finsight.ingestion.chunker import ChunkResult
from finsight.ingestion.embedder import EMBEDDING_DIM, embed_chunks
from finsight.ingestion.indexer import index_document

# ---------------------------------------------------------------------------
# Fake do client OpenAI
# ---------------------------------------------------------------------------


def _make_fake_create(dim: int = EMBEDDING_DIM) -> Any:
    """
    Retorna um substituto async de client.embeddings.create.

    Imita o contrato real que o embedder consome: response.data com itens que
    têm .index e .embedding, e response.usage.total_tokens. Os vetores são
    constantes (0.1) mas com a dimensão correta — o suficiente para o INSERT no
    pgvector e para passar pela validação de dimensão do embedder.
    """

    async def fake_create(*, model: str, input: list[str]) -> Any:
        data = [
            SimpleNamespace(index=i, embedding=[0.1] * dim)
            for i in range(len(input))
        ]
        usage = SimpleNamespace(total_tokens=sum(len(t.split()) for t in input))
        return SimpleNamespace(data=data, usage=usage)

    return fake_create


# ---------------------------------------------------------------------------
# Fixture de sessão isolada (rollback no teardown)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    """
    Sessão async que reverte tudo no fim — isolamento sem TRUNCATE.

    Como index_document não commita, o rollback aqui descarta os flushes do
    teste. Cada teste começa com o banco no mesmo estado.
    """
    async with AsyncSessionLocal() as s:
        try:
            yield s
        finally:
            await s.rollback()


# ---------------------------------------------------------------------------
# Teste
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_embed_and_index_pipeline(
    session: AsyncSession,
    mocker: Any,
) -> None:
    # Mocka o client OpenAI no embedder — sem rede, sem custo.
    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=_make_fake_create())
    )
    mocker.patch(
        "finsight.ingestion.embedder._get_client",
        return_value=fake_client,
    )

    # 1. Chunks de entrada (simulam a saída do chunker).
    chunks = [
        ChunkResult(
            content=f"Trecho do relatório número {i}.",
            chunk_index=i,
            metadata={"ticker": "PETR4", "page_number": 1, "chunk_index_in_page": i},
        )
        for i in range(3)
    ]

    # 2. Embedder: cada chunk ganha um vetor de dimensão correta.
    embedded = await embed_chunks(chunks)
    assert len(embedded) == 3
    assert all(len(e.embedding) == EMBEDDING_DIM for e in embedded)

    # 3. Indexer: grava no Postgres (flush, sem commit).
    document = await index_document(
        session,
        ticker="PETR4",
        title="Relatório de Resultados Q3",
        chunks=embedded,
        source_type="earnings",
    )
    await session.flush()
    assert document.id is not None

    # 4. Lê de volta o Document e confere os campos.
    db_doc = await session.get(Document, document.id)
    assert db_doc is not None
    assert db_doc.ticker == "PETR4"
    assert db_doc.source_type == "earnings"

    # 5. Lê os chunks e valida count, FK, dimensão do vetor e metadados.
    result = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
    )
    rows = result.scalars().all()
    assert len(rows) == 3
    assert [r.chunk_index for r in rows] == [0, 1, 2]
    assert all(len(r.embedding) == EMBEDDING_DIM for r in rows)
    assert rows[0].metadata_ == {
        "ticker": "PETR4",
        "page_number": 1,
        "chunk_index_in_page": 0,
    }


@pytest.mark.integration
async def test_index_document_empty_chunks(
    session: AsyncSession,
) -> None:
    """Documento sem chunks: cria o Document, zero chunks — sem erro."""
    document = await index_document(
        session,
        ticker="VALE3",
        title="Documento vazio",
        chunks=[],
    )
    await session.flush()

    result = await session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == document.id)
    )
    assert result.scalars().all() == []
