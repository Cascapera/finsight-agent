"""
Testes do re-ranking (Passo 3 da Semana 3).

Estratégia de mock — nenhum acesso à rede:
  - _get_ranking_client (reranker): substituído por um fake cujo .ainvoke devolve
    um RerankResult controlado. Dois fakes:
      * _FixedRankingClient: devolve um ranking pré-montado (testes de unidade,
        onde controlamos a lista de chunks e seus índices diretamente).
      * _KeywordRankingClient: LÊ os candidatos do prompt e pontua por prioridade
        de palavra-chave (teste de integração, robusto à ordem que o banco devolve).
  - embeddings (embedder._get_client): fake ORTOGONAL palavra-chave -> dimensão,
    o mesmo dos testes do retriever/HyDE.

A maioria dos testes NÃO toca o banco: `rerank` é a primitiva e recebe chunks
fabricados. Só o teste end-to-end de `retrieve_and_rerank` precisa do Postgres.
"""

import re
import uuid
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
from finsight.retrieval import reranker
from finsight.retrieval.reranker import RankedCandidate, RerankResult
from finsight.retrieval.retriever import RetrievedChunk

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FixedRankingClient:
    """Devolve sempre o mesmo RerankResult, ignorando o prompt."""

    def __init__(self, result: RerankResult) -> None:
        self._result = result

    async def ainvoke(self, messages: Any, **kwargs: Any) -> RerankResult:
        return self._result


class _ExplodingRankingClient:
    """Falha se chamado — usado para provar que os guards NÃO invocam o LLM."""

    async def ainvoke(self, messages: Any, **kwargs: Any) -> RerankResult:
        raise AssertionError("o LLM não deveria ter sido chamado neste caminho")


class _KeywordRankingClient:
    """
    Juiz fake determinístico: lê os candidatos do prompt e pontua por prioridade
    de palavra-chave. Robusto à ordem em que o banco devolve os candidatos —
    pontua pelo CONTEÚDO, não pelo índice posicional fixo.
    """

    def __init__(self, priority: list[str]) -> None:
        # priority[0] é o mais relevante; cada posição abaixo perde 2 pontos.
        self._priority = priority

    async def ainvoke(self, messages: Any, **kwargs: Any) -> RerankResult:
        human = messages[-1].content
        # Cada candidato foi formatado como "[i] conteúdo" numa linha. `.` não
        # cruza '\n' por padrão, então cada match captura um candidato.
        pairs = re.findall(r"\[(\d+)\]\s*(.*)", human)
        ranking = []
        for idx_str, text in pairs:
            lowered = text.lower()
            relevance = 0.0
            for rank, keyword in enumerate(self._priority):
                if keyword in lowered:
                    relevance = max(0.0, 10.0 - 2.0 * rank)
                    break
            ranking.append(RankedCandidate(index=int(idx_str), relevance=relevance))
        return RerankResult(ranking=ranking)


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
    return _unit(3)  # neutro


def _make_fake_embeddings_create() -> Any:
    async def fake_create(*, model: str, input: list[str]) -> Any:
        data = [SimpleNamespace(index=i, embedding=_embedding_for(t)) for i, t in enumerate(input)]
        usage = SimpleNamespace(total_tokens=sum(len(t.split()) for t in input))
        return SimpleNamespace(data=data, usage=usage)

    return fake_create


def _chunk(content: str, score: float, chunk_index: int = 0) -> RetrievedChunk:
    """Fabrica um RetrievedChunk com cosine `score` (sem tocar o banco)."""
    return RetrievedChunk(
        content=content,
        score=score,
        document_id=uuid.uuid4(),
        document_title="doc",
        chunk_index=chunk_index,
        metadata={},
    )


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


async def test_rerank_reorders_by_llm_scores(mocker: Any) -> None:
    """O juiz inverte a ordem do cosine -> resultado sai na ordem do juiz."""
    chunks = [
        _chunk("A lucro", 0.7, 0),
        _chunk("B dívida", 0.6, 1),
        _chunk("C caixa", 0.5, 2),
    ]
    # Cosine ordenava [A, B, C]; o juiz prefere C > A > B.
    result = RerankResult(
        ranking=[
            RankedCandidate(index=2, relevance=9.0),
            RankedCandidate(index=0, relevance=5.0),
            RankedCandidate(index=1, relevance=1.0),
        ]
    )
    mocker.patch.object(reranker, "_get_ranking_client", return_value=_FixedRankingClient(result))

    out = await reranker.rerank("q", chunks, top_n=3)

    assert [c.content for c in out] == ["C caixa", "A lucro", "B dívida"]
    # Score reescrito com a relevância do juiz normalizada (rel/10).
    assert [c.score for c in out] == pytest.approx([0.9, 0.5, 0.1])


async def test_rerank_rewrites_score_and_preserves_cosine(mocker: Any) -> None:
    """`score` vira a relevância do juiz; o cosine original vai para metadata."""
    chunks = [_chunk("A lucro", 0.42, 0), _chunk("B caixa", 0.31, 1)]
    result = RerankResult(
        ranking=[
            RankedCandidate(index=1, relevance=8.0),
            RankedCandidate(index=0, relevance=2.0),
        ]
    )
    mocker.patch.object(reranker, "_get_ranking_client", return_value=_FixedRankingClient(result))

    out = await reranker.rerank("q", chunks, top_n=2)

    assert out[0].content == "B caixa"
    assert out[0].score == pytest.approx(0.8)
    # Cosine original de "B caixa" preservado para comparação.
    assert out[0].metadata["vector_score"] == pytest.approx(0.31)
    assert out[1].metadata["vector_score"] == pytest.approx(0.42)


async def test_rerank_truncates_to_top_n(mocker: Any) -> None:
    """Com 3 candidatos e top_n=2, sobram os 2 melhores segundo o juiz."""
    chunks = [_chunk("A", 0.9, 0), _chunk("B", 0.8, 1), _chunk("C", 0.7, 2)]
    result = RerankResult(
        ranking=[
            RankedCandidate(index=2, relevance=9.0),
            RankedCandidate(index=1, relevance=7.0),
            RankedCandidate(index=0, relevance=3.0),
        ]
    )
    mocker.patch.object(reranker, "_get_ranking_client", return_value=_FixedRankingClient(result))

    out = await reranker.rerank("q", chunks, top_n=2)

    assert [c.content for c in out] == ["C", "B"]


async def test_rerank_robust_to_bad_indices(mocker: Any) -> None:
    """
    O juiz alucina um índice (99), duplica um (2) e omite outro (1). Nenhum chunk
    se perde: índices inválidos/duplicados são ignorados e o omitido vai pro fim,
    na ordem original.
    """
    chunks = [_chunk("A", 0.9, 0), _chunk("B", 0.6, 1), _chunk("C", 0.5, 2)]
    result = RerankResult(
        ranking=[
            RankedCandidate(index=2, relevance=8.0),
            RankedCandidate(index=99, relevance=10.0),  # alucinação -> ignorado
            RankedCandidate(index=2, relevance=5.0),  # duplicata -> ignorada
            RankedCandidate(index=0, relevance=3.0),
            # índice 1 nunca aparece -> omitido
        ]
    )
    mocker.patch.object(reranker, "_get_ranking_client", return_value=_FixedRankingClient(result))

    out = await reranker.rerank("q", chunks, top_n=5)

    # Pontuados [C, A] na ordem do juiz; omitido [B] anexado no fim.
    assert [c.content for c in out] == ["C", "A", "B"]
    assert len(out) == 3  # nada se perdeu
    # O omitido mantém o cosine como score (sem nota do juiz para inventar).
    assert out[2].content == "B"
    assert out[2].score == pytest.approx(0.6)
    assert out[2].metadata["vector_score"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("query", "chunks"),
    [
        ("q", []),  # lista vazia
        ("q", [_chunk("só um", 0.5, 0)]),  # candidato único
        ("", [_chunk("a", 0.5, 0), _chunk("b", 0.4, 1)]),  # query vazia
    ],
)
async def test_rerank_guards_skip_llm(
    query: str, chunks: list[RetrievedChunk], mocker: Any
) -> None:
    """Lista vazia, candidato único ou query vazia não devem chamar o LLM."""
    mocker.patch.object(reranker, "_get_ranking_client", return_value=_ExplodingRankingClient())

    out = await reranker.rerank(query, chunks, top_n=5)

    assert out == chunks  # devolve o que recebeu, sem reordenar


# ---------------------------------------------------------------------------
# Teste de integração: over-fetch -> rerank muda a ordem da busca vetorial
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_retrieve_and_rerank_overrides_vector_order(
    session: AsyncSession, mocker: Any
) -> None:
    """
    A busca vetorial (query 'caixa') colocaria o chunk de caixa em 1º; o juiz
    prioriza 'lucro' e promove o chunk do lucro — provando que o re-rank
    sobrepõe a ordem do estágio vetorial.
    """
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=_make_fake_embeddings_create()))
    mocker.patch("finsight.ingestion.embedder._get_client", return_value=fake_client)
    mocker.patch.object(
        reranker,
        "_get_ranking_client",
        return_value=_KeywordRankingClient(priority=["lucro", "caixa", "dívida"]),
    )

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

    out = await reranker.retrieve_and_rerank(
        "qual a situação de caixa?",  # casa com o chunk de caixa no estágio vetorial
        ticker="PETR4",
        fetch_k=3,
        top_n=2,
        session=session,
    )

    assert len(out) == 2
    # O juiz promoveu o lucro, apesar de o vetor o ter posto em último (sim 0).
    assert "lucro" in out[0].content.lower()
    # O cosine baixo do lucro (ortogonal à query 'caixa') foi preservado.
    assert out[0].metadata["vector_score"] == pytest.approx(0.0)
    assert out[0].score == pytest.approx(1.0)  # relevância máxima do juiz
