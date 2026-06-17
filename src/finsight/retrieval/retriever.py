"""
Retrieval base: busca por similaridade vetorial no pgvector.

Esta é a fundação do RAG. O fluxo de leitura é o espelho da ingestão:

    ingestão:  PDF -> chunks -> embeddings -> INSERT no pgvector
    retrieval: query -> embedding -> SELECT por similaridade -> chunks

Tudo aqui é infraestrutura "pura": recebe uma query, devolve chunks ordenados
por relevância. As camadas mais inteligentes da Semana 3 — HyDE e re-ranking —
se constroem EM CIMA desta função, não dentro dela. Manter o retriever simples
e previsível é o que torna possível medir o ganho de cada técnica isoladamente.

Decisão de erro: este módulo PROPAGA exceções (não captura). Quem traduz erro em
`state["errors"]` é o nó RAG do grafo (Semana 6) — a regra "erro no nó" do
projeto vive na camada de orquestração, não na de infra.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finsight.db.models import Document, DocumentChunk
from finsight.db.session import AsyncSessionLocal
from finsight.graph.state import RAGOutput
from finsight.ingestion.embedder import embed_texts

# top_k default: quantos chunks a busca vetorial devolve.
# 5 é um bom ponto de partida para ir direto ao contexto do LLM. Quando o
# re-ranking entrar (Passo 3), vamos buscar MAIS aqui (ex. 20) e deixar o
# re-ranker reduzir para os 5 melhores — por isso top_k é parâmetro, não fixo.
DEFAULT_TOP_K = 5


@dataclass
class RetrievedChunk:
    """
    Um chunk recuperado do banco, já com seu score de similaridade.

    Por que um tipo próprio e não devolver `RAGOutput` direto?
    O RAGOutput é o formato "achatado" que vai pro state do grafo (listas
    paralelas de strings/floats). Aqui no meio do pipeline precisamos de um
    objeto rico: HyDE e re-ranking trabalham chunk-a-chunk, reordenam, cortam.
    Carregar document_id/título/metadados junkos evita uma segunda query depois.
    A conversão para RAGOutput é feita só no fim, por `to_rag_output`.
    """

    content: str
    score: float  # similaridade de cosseno em [0, 1] (quanto maior, mais relevante)
    document_id: uuid.UUID
    document_title: str
    chunk_index: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@asynccontextmanager
async def _session_scope(session: AsyncSession | None) -> AsyncIterator[AsyncSession]:
    """
    Padrão "owned-or-borrowed" para a sessão do banco.

    - session=None  -> abrimos uma sessão PRÓPRIA e a fechamos no fim (caminho de
                       produção: o retriever é auto-suficiente).
    - session=<dada> -> usamos a sessão EMPRESTADA e NÃO a fechamos (quem criou,
                       fecha). É o que permite o teste injetar a sessão do fixture
                       de rollback — lemos dados só flushados, sem commit — e o que
                       deixará o nó RAG compartilhar a mesma transação.
    """
    if session is not None:
        yield session
    else:
        async with AsyncSessionLocal() as owned:
            yield owned


async def retrieve(
    query: str,
    *,
    ticker: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    session: AsyncSession | None = None,
) -> list[RetrievedChunk]:
    """
    Recupera os `top_k` chunks mais similares à `query`.

    Args:
        query: pergunta do usuário em linguagem natural.
        ticker: se informado, restringe a busca aos documentos daquele ativo
                (JOIN com documents.ticker). É keyword-only (`*`) de propósito:
                força o call-site a escrever `retrieve(q, ticker="PETR4")`, que
                é auto-documentado, em vez de um segundo posicional ambíguo.
        top_k: quantos chunks devolver.
        session: sessão a reusar (emprestada). Se None, abre a própria. Ver
                 _session_scope.

    Returns:
        Lista de RetrievedChunk ORDENADA do mais relevante para o menos.
        Lista vazia se a query for vazia ou nada casar.
    """
    # Guard clause: query vazia não tem o que buscar. Evita gastar uma chamada
    # de embedding (custo) e uma query ao banco para retornar nada.
    if not query.strip():
        return []

    # 1) Embedar a query com o MESMO modelo da ingestão.
    # embed_texts recebe e devolve listas (é batch por natureza); passamos uma
    # query e pegamos o primeiro (e único) vetor. Reusar esta função — em vez de
    # chamar a OpenAI aqui de novo — garante dois invariantes de graça:
    #   (a) a query é embedada exatamente como os chunks foram (mesmo modelo,
    #       mesma validação de dimensão 1536);
    #   (b) é o mesmo ponto que os testes mockam — o retriever herda o mock.
    query_embedding = (await embed_texts([query]))[0]

    # 2) Montar a busca por similaridade.
    #
    # `.cosine_distance(vec)` é um método que o pgvector injeta na coluna Vector.
    # Ele gera o operador `<=>` do Postgres — DISTÂNCIA de cosseno, não
    # similaridade. Relação: distância = 1 - similaridade.
    #   - distância 0   -> vetores idênticos     -> similaridade 1
    #   - distância 1   -> ortogonais            -> similaridade 0
    # Por isso ordenamos por distância ASCENDENTE (menor distância = mais perto)
    # e convertemos para similaridade ao montar o resultado.
    #
    # Damos um .label() para conseguir referenciar a mesma expressão no SELECT e
    # no ORDER BY sem recalcular — e para extrair o valor da Row depois.
    distance = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        # Selecionamos a entidade chunk, a entidade document (para o título) e a
        # distância calculada. Três "colunas" -> cada Row vem como (chunk, doc, dist).
        select(DocumentChunk, Document, distance)
        # JOIN explícito chunk -> documento. Precisamos do Document para o título
        # (fonte do RAGOutput.sources) e para o filtro por ticker.
        .join(Document, DocumentChunk.document_id == Document.id)
        # Só chunks que TÊM embedding. A coluna é nullable (um chunk pode ter
        # sido inserido antes de embedar); o operador `<=>` quebra com NULL.
        .where(DocumentChunk.embedding.is_not(None))
        # ORDER BY distância: o índice ivfflat (vector_cosine_ops) acelera
        # justamente esta ordenação. Sem o ORDER BY casando com o índice, o
        # Postgres faria full scan + sort.
        .order_by(distance)
        # LIMIT no banco — nunca traga 10k linhas para cortar no Python.
        .limit(top_k)
    )

    # Filtro por ativo, aplicado condicionalmente. SQLAlchemy é imutável aqui:
    # `.where()` devolve um novo statement, não muta o anterior.
    if ticker is not None:
        stmt = stmt.where(Document.ticker == ticker)

    # 3) Executar dentro de uma sessão async (própria ou emprestada — ver
    # _session_scope). Como é só leitura, não há commit.
    async with _session_scope(session) as s:
        result = await s.execute(stmt)
        rows = result.all()

    # 4) Traduzir Rows -> RetrievedChunk, convertendo distância em similaridade.
    return [
        RetrievedChunk(
            content=chunk.content,
            # 1 - distância. Embeddings da OpenAI são normalizados (norma 1), então
            # a similaridade de cosseno fica em [0, 1] na prática para texto
            # relacionado; valores negativos seriam textos "opostos" e são raros.
            score=1.0 - distance_value,
            document_id=chunk.document_id,
            document_title=doc.title,
            chunk_index=chunk.chunk_index,
            metadata=chunk.metadata_ or {},
        )
        for chunk, doc, distance_value in rows
    ]


def to_rag_output(chunks: list[RetrievedChunk]) -> RAGOutput:
    """
    Converte a lista rica de RetrievedChunk no RAGOutput achatado do state.

    Ponte fina entre a camada de retrieval (objetos ricos) e o grafo (listas
    paralelas). Mantemos isso separado de `retrieve` porque o re-ranking vai
    se inserir ENTRE os dois: retrieve -> [re-rank] -> to_rag_output. Se a
    conversão estivesse embutida no retrieve, não haveria onde encaixar o
    re-ranker sem desfazer o RAGOutput.

    As três listas são paralelas por índice: chunks[i], sources[i] e
    relevance_scores[i] descrevem o mesmo trecho.
    """
    return RAGOutput(
        chunks=[c.content for c in chunks],
        sources=[c.document_title for c in chunks],
        relevance_scores=[c.score for c in chunks],
    )
