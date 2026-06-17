"""
HyDE — Hypothetical Document Embeddings.

Problema: a pergunta do usuário (curta, interrogativa) e o trecho-alvo do
relatório (longo, afirmativo, técnico) vivem em regiões diferentes do espaço de
embeddings, mesmo falando do mesmo assunto. Buscar com a pergunta crua perde
recall.

Solução: pedir ao LLM um "documento hipotético" — um trecho que PARECERIA a
resposta — e buscar com o embedding DELE. O documento hipotético tem a forma de
um documento real, então cai perto dos chunks reais relevantes.

A alucinação do LLM aqui é esperada e inofensiva: nunca mostramos o documento
hipotético ao usuário; usamos apenas a DIREÇÃO semântica do embedding dele para
achar chunks REAIS. A resposta final é construída sobre os chunks reais.

Por que langchain (ChatOpenAI) e não o SDK cru? Coerente com a regra do
embedder.py: SDK cru para ingestão offline; langchain para os passos de
RACIOCÍNIO do agente, onde o tracing do LangSmith (Semana 7) agrega valor.
"""

import asyncio
import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from finsight.db.session import settings
from finsight.ingestion.embedder import embed_texts
from finsight.retrieval.retriever import (
    DEFAULT_TOP_K,
    RetrievedChunk,
    search_by_embedding,
)

logger = logging.getLogger(__name__)

# O prompt define o papel e, crucialmente, o ESTILO: queremos texto com cara de
# relatório (afirmativo, com métricas), não uma resposta conversacional. Pedimos
# explicitamente para NÃO ressalvar/“é hipotético” — qualquer meta-texto polui o
# embedding com tokens que não existem nos documentos reais.
_HYDE_SYSTEM_PROMPT = (
    "Você é um analista financeiro sênior. Dada uma pergunta, escreva um trecho "
    "conciso (3 a 5 frases), em português, redigido no estilo de um relatório "
    "financeiro corporativo, que responderia diretamente a essa pergunta. Seja "
    "específico no tom (cite métricas, períodos e rubricas como um relatório real "
    "faria). Não diga que o trecho é hipotético, não faça ressalvas e não repita "
    "a pergunta — escreva apenas o trecho, como se fosse extraído do documento."
)


@lru_cache(maxsize=1)
def _get_chat_client() -> ChatOpenAI:
    """
    Client de chat, criado uma única vez (lru_cache). Ponto de mock nos testes.

    temperature=0.7: queremos DIVERSIDADE entre as amostras quando n_samples>1 —
    é o que dá robustez à média de embeddings (gerações diferentes, erros que se
    cancelam). max_retries=4: o SDK reintenta 429/5xx com backoff.
    """
    return ChatOpenAI(
        # SecretStr: o tipo que o ChatOpenAI espera; evita a chave aparecer em
        # reprs/tracebacks (str crua vazaria).
        api_key=SecretStr(settings.openai_api_key),
        model=settings.active_llm_model,
        temperature=0.7,
        max_retries=4,
        timeout=60.0,
    )


async def generate_hypothetical_documents(query: str, *, n_samples: int = 1) -> list[str]:
    """
    Gera `n_samples` documentos hipotéticos que responderiam à `query`.

    As chamadas são concorrentes (asyncio.gather): n_samples requisições em
    paralelo custam ~o tempo de uma só, não n vezes. Cada uma usa o mesmo prompt;
    a variação vem da temperatura > 0.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples deve ser >= 1, recebido {n_samples}.")

    client = _get_chat_client()
    messages = [
        SystemMessage(content=_HYDE_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    async def _one() -> str:
        response = await client.ainvoke(messages)
        # response.content pode ser str ou list[str|dict] (multimodal). Para texto
        # puro é str; normalizamos defensivamente para nunca embedar um repr.
        content = response.content
        return content if isinstance(content, str) else str(content)

    docs = list(await asyncio.gather(*[_one() for _ in range(n_samples)]))
    logger.debug("HyDE gerou %d documento(s) para a query %r", len(docs), query)
    return docs


def _mean_vectors(vectors: list[list[float]]) -> list[float]:
    """
    Média elemento-a-elemento de vetores (a "direção média").

    A média de várias gerações cancela os erros idiossincráticos de cada uma e
    preserva a direção comum (o tópico de fato). Vetores não-unitários não são
    problema: o cosseno normaliza internamente, só a direção importa.
    """
    if not vectors:
        raise ValueError("Sem vetores para promediar.")
    n = len(vectors)
    # zip(*vectors) percorre coluna a coluna (dimensão a dimensão); strict=True
    # garante que todos os vetores tenham a mesma dimensão.
    return [sum(column) / n for column in zip(*vectors, strict=True)]


async def hyde_embedding(
    query: str,
    *,
    n_samples: int = 1,
    include_query: bool = True,
) -> list[float]:
    """
    Calcula o embedding HyDE: média dos embeddings dos documentos hipotéticos.

    include_query=True adiciona o embedding da PERGUNTA original à média — uma
    âncora: se as gerações derivarem do tópico, a query puxa o vetor de volta.
    """
    docs = await generate_hypothetical_documents(query, n_samples=n_samples)

    texts = [*docs, query] if include_query else docs
    vectors = await embed_texts(texts)
    return _mean_vectors(vectors)


async def hyde_retrieve(
    query: str,
    *,
    ticker: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    n_samples: int = 1,
    include_query: bool = True,
    session: AsyncSession | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieval com HyDE: gera doc(s) hipotético(s), embeda, busca por similaridade.

    É o irmão "esperto" de `retrieve`: mesma assinatura de busca (ticker, top_k,
    session), mas o vetor de consulta vem do documento hipotético em vez da query
    crua. Compor sobre `search_by_embedding` é o que torna isso trivial.
    """
    if not query.strip():
        return []

    embedding = await hyde_embedding(query, n_samples=n_samples, include_query=include_query)
    return await search_by_embedding(embedding, ticker=ticker, top_k=top_k, session=session)
