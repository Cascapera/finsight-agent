"""
Persistência dos chunks no PostgreSQL/pgvector.

Responsabilidade: receber os EmbeddedChunk (já com vetor) produzidos pelo
embedder, criar o Document pai e gravar os DocumentChunk filhos numa única
transação atômica.

Decisão de design: a função recebe a AsyncSession e NÃO dá commit — quem
controla a transação é o chamador (padrão unit of work). Isso mantém o indexer
testável e componível: o mesmo `index_document` serve tanto para um script de
ingestão quanto para um endpoint que já tem uma sessão aberta.

Idempotência: por ora NÃO há deduplicação — re-ingerir o mesmo PDF cria um
Document novo. A dedup (via content_hash) fica para uma etapa futura, quando o
pipeline de orquestração existir.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from finsight.db.models import Document, DocumentChunk
from finsight.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)


async def index_document(
    session: AsyncSession,
    *,
    ticker: str,
    title: str,
    chunks: list[EmbeddedChunk],
    source_type: str | None = None,
    source_url: str | None = None,
) -> Document:
    """
    Persiste um documento e seus chunks com embedding.

    O `*` força todos os argumentos de dados a serem nomeados — com vários
    parâmetros `str | None` opcionais, isso impede trocar source_type por
    source_url posicionalmente por engano.

    Args:
        session: sessão async ABERTA. O commit é responsabilidade do chamador.
        ticker: código do ativo (ex: "PETR4").
        title: título do documento — aparece nas citações do RAG.
        chunks: chunks já embeddados (saída de embed_chunks).
        source_type: categoria ("earnings", "ri", "fii_report").
        source_url: URL de origem, se houver.

    Returns:
        O Document persistido, com `id` já populado pelo banco.

    Uso típico (o chamador controla a transação):
        async with AsyncSessionLocal() as session:
            doc = await index_document(session, ticker="PETR4", title=..., chunks=...)
            await session.commit()
    """
    document = Document(
        ticker=ticker,
        title=title,
        source_type=source_type,
        source_url=source_url,
    )
    session.add(document)

    # flush != commit. O flush envia o INSERT e popula document.id (gerado pelo
    # servidor via gen_random_uuid, retornado por RETURNING) sem fechar a
    # transação. Precisamos do id agora para usá-lo como FK nos chunks, mas só
    # confirmamos tudo no commit do chamador — atomicamente.
    await session.flush()

    # add_all (ORM) em vez de insert() Core: o volume aqui é dezenas de PDFs
    # (o próprio models.py assume esse regime ao escolher ivfflat sobre hnsw),
    # então legibilidade vence o ganho marginal de throughput do Core — e
    # evitamos a pegadinha de nome metadata_ vs "metadata".
    session.add_all(
        [
            DocumentChunk(
                document_id=document.id,
                content=chunk.content,
                # embedding é list[float]; o tipo Vector do pgvector faz o bind
                embedding=chunk.embedding,
                chunk_index=chunk.chunk_index,
                # atributo metadata_ -> coluna "metadata" (ver db/models.py)
                metadata_=chunk.metadata,
            )
            for chunk in chunks
        ]
    )
    await session.flush()

    logger.info(
        "Documento indexado: ticker=%s title=%r chunks=%d id=%s",
        ticker,
        title,
        len(chunks),
        document.id,
    )

    return document
