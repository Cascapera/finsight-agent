"""
Testes do HyDE: o documento hipotético (não a query crua) guia a busca.

Requer Postgres com pgvector (para o teste marcado @integration):
    docker compose up -d postgres
    alembic upgrade head

Estratégia de mock — dois clients falsos, nenhum acesso à rede:
  - chat (hyde._get_chat_client): devolve um documento hipotético controlado.
  - embeddings (embedder._get_client): fake ORTOGONAL palavra-chave -> dimensão,
    o mesmo do teste do retriever.

A montagem é proposital: a QUERY não contém nenhuma palavra-chave (vira um vetor
neutro, dim 3), mas o DOCUMENTO HIPOTÉTICO contém "lucro" (dim 0). Se a busca
acertar o chunk do "lucro", é prova de que foi o documento gerado — e não a query
crua — que direcionou o retrieval. Esse é o coração do HyDE.
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
from finsight.retrieval import hyde

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_KEYWORD_DIM = {"lucro": 0, "dívida": 1, "caixa": 2}


def _unit(index: int, dim: int = EMBEDDING_DIM) -> list[float]:
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


def _embedding_for(text: str) -> list[float]:
    lowered = text.lower()
    for keyword, index in _KEYWORD_DIM.items():
        if keyword in lowered:
            return _unit(index)
    return _unit(3)  # neutro: ortogonal a lucro/dívida/caixa


def _make_fake_embeddings_create() -> Any:
    async def fake_create(*, model: str, input: list[str]) -> Any:
        data = [SimpleNamespace(index=i, embedding=_embedding_for(t)) for i, t in enumerate(input)]
        usage = SimpleNamespace(total_tokens=sum(len(t.split()) for t in input))
        return SimpleNamespace(data=data, usage=usage)

    return fake_create


class _FakeChat:
    """Substituto de ChatOpenAI: .ainvoke devolve sempre o mesmo documento."""

    def __init__(self, document: str) -> None:
        self._document = document

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(content=self._document)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as s:
        try:
            yield s
        finally:
            await s.rollback()


# ---------------------------------------------------------------------------
# Testes que NÃO tocam o banco
# ---------------------------------------------------------------------------


async def test_generate_documents_respects_n_samples(mocker: Any) -> None:
    """n_samples=3 => três gerações (chamadas concorrentes)."""
    mocker.patch.object(
        hyde, "_get_chat_client", return_value=_FakeChat("O lucro líquido cresceu.")
    )

    docs = await hyde.generate_hypothetical_documents("pergunta qualquer", n_samples=3)

    assert len(docs) == 3
    assert all("lucro" in d.lower() for d in docs)


async def test_generate_documents_rejects_zero_samples(mocker: Any) -> None:
    """n_samples < 1 é erro de programação — falha cedo."""
    mocker.patch.object(hyde, "_get_chat_client", return_value=_FakeChat("x"))

    with pytest.raises(ValueError, match="n_samples"):
        await hyde.generate_hypothetical_documents("q", n_samples=0)


async def test_hyde_embedding_averages_doc_and_query(mocker: Any) -> None:
    """
    Com include_query=True, o embedding final é a média do doc (dim 0) e da
    query neutra (dim 3): 0.5 em cada uma dessas posições.
    """
    mocker.patch.object(hyde, "_get_chat_client", return_value=_FakeChat("lucro recorde"))
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=_make_fake_embeddings_create()))
    mocker.patch("finsight.ingestion.embedder._get_client", return_value=fake_client)

    # query SEM palavra-chave -> dim 3
    vec = await hyde.hyde_embedding("como vai a empresa", n_samples=1, include_query=True)

    assert vec[0] == pytest.approx(0.5)  # contribuição do doc (lucro)
    assert vec[3] == pytest.approx(0.5)  # contribuição da query neutra
    assert vec[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Teste de integração: HyDE muda o resultado da busca
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_hyde_retrieve_is_driven_by_generated_doc(session: AsyncSession, mocker: Any) -> None:
    """A query crua é neutra, mas o doc hipotético tem 'lucro' -> chunk do lucro vence."""
    # O LLM "responde" com um trecho sobre lucro.
    mocker.patch.object(
        hyde,
        "_get_chat_client",
        return_value=_FakeChat("O lucro líquido da companhia cresceu 20% no período."),
    )
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=_make_fake_embeddings_create()))
    mocker.patch("finsight.ingestion.embedder._get_client", return_value=fake_client)

    # Semeia 3 chunks ortogonais.
    chunks = [
        ChunkResult(content=c, chunk_index=i, metadata={"ticker": "PETR4"})
        for i, c in enumerate(
            [
                "O lucro líquido cresceu 20% no trimestre.",  # dim 0
                "A dívida bruta aumentou frente ao ano anterior.",  # dim 1
                "O fluxo de caixa livre ficou positivo.",  # dim 2
            ]
        )
    ]
    embedded = await embed_chunks(chunks)
    doc = await index_document(
        session, ticker="PETR4", title="Relatório PETR4", chunks=embedded, source_type="earnings"
    )
    await session.flush()
    assert doc.id is not None

    # Query propositalmente VAGA (sem 'lucro'/'dívida'/'caixa').
    results = await hyde.hyde_retrieve(
        "como a empresa se saiu no período?",
        ticker="PETR4",
        top_k=3,
        session=session,
    )

    assert len(results) == 3
    # O documento hipotético falava de lucro -> o chunk do lucro vem primeiro.
    assert "lucro" in results[0].content.lower()
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
