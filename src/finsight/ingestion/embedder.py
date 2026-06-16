"""
Geração de embeddings para os chunks de documentos.

Responsabilidade: receber os ChunkResult produzidos pelo chunker, gerar o
vetor de embedding de cada um via OpenAI (text-embedding-3-small) e devolver
EmbeddedChunk — o tipo que o indexer consome para o INSERT no pgvector.

Por que SDK `openai` direto e não `OpenAIEmbeddings` do langchain?
Ingestão é um processo offline em batch. Aqui queremos controle explícito de
batch size, custo e retry — coisas que a abstração do langchain esconde. O
langchain fica reservado para os nós do grafo, onde o tracing automático do
LangSmith de fato agrega valor.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from openai import AsyncOpenAI

from finsight.db.session import settings
from finsight.ingestion.chunker import ChunkResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes do modelo de embedding
# ---------------------------------------------------------------------------

# Dimensão fixa do text-embedding-3-small. Precisa bater com Vector(1536) em
# db/models.py — se divergir, o operador de similaridade do pgvector quebra em
# runtime. Validamos cada vetor contra esta constante para falhar cedo (na
# ingestão) em vez de tarde (na busca).
EMBEDDING_DIM = 1536

# Preço do text-embedding-3-small: US$ 0.02 por 1M de tokens (jun/2026).
# Usado só para logar o custo real da ingestão — não há cobrança extra do nosso
# lado, apenas observabilidade de gasto.
COST_PER_1M_TOKENS_USD = 0.02


# ---------------------------------------------------------------------------
# Tipo de saída
# ---------------------------------------------------------------------------


@dataclass
class EmbeddedChunk:
    """
    Um ChunkResult que já passou pelo embedder.

    Compomos em vez de adicionar `embedding` ao ChunkResult porque as etapas
    são distintas no sistema de tipos: receber um EmbeddedChunk garante, em
    tempo de compilação (mypy), que o chunk tem vetor — o indexer não precisa
    checar `if embedding is None`.
    """

    content: str
    chunk_index: int
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_client() -> AsyncOpenAI:
    """
    Retorna o client OpenAI async — criado uma única vez (lru_cache).

    max_retries=4: o próprio SDK reintenta em 429 (rate limit) e 5xx com
    backoff exponencial + jitter. Não usamos tenacity — seria reinventar o que
    o SDK já faz de forma battle-tested. Subimos de 2 (default) para 4 porque
    ingestão em batch tende a esbarrar em rate limit com mais frequência.

    timeout=60s: requests de embedding em batch são maiores que um chat comum;
    o default de alguns SDKs é curto demais para lotes de 100 chunks longos.
    """
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        max_retries=4,
        timeout=60.0,
    )


# ---------------------------------------------------------------------------
# Núcleo: textos -> vetores
# ---------------------------------------------------------------------------


def _batched(items: list[str], size: int) -> list[list[str]]:
    """Fatia uma lista em lotes de no máximo `size` elementos."""
    return [items[i : i + size] for i in range(0, len(items), size)]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos, preservando a ordem.

    Esta é a função-núcleo, pura em termos de domínio (entra texto, sai vetor).
    É o ponto de mock no CI: os testes substituem `_get_client` ou esta função
    para não bater na API real da OpenAI.

    Batching: a API aceita uma lista de inputs por request — mandamos lotes de
    `settings.embedding_batch_size` (default 100) em vez de 1 request por texto.
    Isso reduz latência e overhead de rede em ordens de grandeza.

    Returns:
        Lista de vetores na MESMA ordem dos textos de entrada. A API garante
        que `response.data` volta na ordem do input (campo `index`), mas
        reordenamos defensivamente por `index` para não depender disso.
    """
    if not texts:
        return []

    client = _get_client()
    model = settings.openai_embedding_model

    all_embeddings: list[list[float]] = []
    total_tokens = 0

    for batch in _batched(texts, settings.embedding_batch_size):
        response = await client.embeddings.create(model=model, input=batch)

        # Reordena por `index` — defensivo. A API normalmente devolve em ordem,
        # mas depender de ordem implícita de I/O é uma fonte clássica de bug
        # silencioso (embedding do chunk A acaba no chunk B).
        ordered = sorted(response.data, key=lambda item: item.index)

        for item in ordered:
            vector = item.embedding
            # Falha cedo se a dimensão divergir do esperado (ex: alguém trocou
            # o modelo no .env sem re-migrar a coluna Vector(1536)).
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding com dimensão {len(vector)}, esperado {EMBEDDING_DIM}. "
                    f"Modelo configurado: {model!r}. Verifique OPENAI_EMBEDDING_MODEL "
                    f"e a dimensão da coluna em db/models.py."
                )
            all_embeddings.append(vector)

        total_tokens += response.usage.total_tokens

    cost_usd = total_tokens / 1_000_000 * COST_PER_1M_TOKENS_USD
    logger.info(
        "Embeddings gerados: %d textos, %d tokens, custo ~US$ %.6f (modelo=%s)",
        len(texts),
        total_tokens,
        cost_usd,
        model,
    )

    return all_embeddings


# ---------------------------------------------------------------------------
# Orquestração: chunks -> chunks com embedding
# ---------------------------------------------------------------------------


async def embed_chunks(chunks: list[ChunkResult]) -> list[EmbeddedChunk]:
    """
    Gera o embedding de cada ChunkResult e devolve EmbeddedChunk.

    Extrai o texto de cada chunk, manda tudo para embed_texts (que cuida do
    batching) e remonta os EmbeddedChunk preservando metadados e índice.

    O `zip` é seguro porque embed_texts garante ordem e cardinalidade iguais à
    entrada — mas usamos strict=True para que uma eventual divergência de
    tamanho exploda em vez de truncar silenciosamente.
    """
    if not chunks:
        return []

    texts = [chunk.content for chunk in chunks]
    embeddings = await embed_texts(texts)

    return [
        EmbeddedChunk(
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            embedding=embedding,
            metadata=chunk.metadata,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
