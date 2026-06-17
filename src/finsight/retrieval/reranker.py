"""
Re-ranking — segundo estágio do retrieval (over-fetch -> reordenar -> top_n).

A busca vetorial (Passo 1) é um BI-ENCODER: query e chunk são embedados
SEPARADAMENTE, então a relevância é só a distância entre dois vetores que nunca
se "viram". Isso dá ótimo recall (acha o conjunto certo) mas ordenação grosseira
(a ordem fina dentre os top candidatos é ruidosa). E essa ordem importa: só uns
top_n chunks entram no contexto do LLM; se o decisivo está na posição 8 e
cortamos no 5, a resposta degrada.

Estratégia (over-fetch):

    query -> busca vetorial barata -> fetch_k=20 candidatos -> rerank -> top_n=5

Buscamos DE PROPÓSITO mais do que vamos usar (20 em vez de 5), aceitando que a
ordem está ruidosa, e aplicamos um scorer mais caro e mais preciso só nesses 20.
O re-ranker é caro POR PAR (query, doc), mas roda em 20 pares, não no corpus
inteiro — a busca vetorial já eliminou 99,9% do corpus barato.

Aqui usamos LLM-as-judge LISTWISE: mandamos a query + os candidatos numerados de
uma vez e o LLM devolve uma relevância por candidato. Vantagens no FinSight:
reusa o ChatOpenAI da stack (ZERO dependência nova), é mockável (CI sem rede) e
devolve structured output tipado (convenção "sem parse manual de texto"). O
custo honesto é uma chamada de LLM no caminho crítico — aceitável para um agente
de análise (qualidade > latência de ms, QPS moderado).

>>> GANCHO DIDÁTICO — alternativa cross-encoder <<<
Um CROSS-ENCODER (ex. ms-marco-MiniLM, Cohere/Voyage Rerank) concatena
[query [SEP] doc] e passa os dois JUNTOS por um Transformer, com atenção cruzada
token-a-token. É mais preciso e mais barato POR QUERY que um LLM, mas exige
modelo dedicado/serviço externo. Ele entraria EXATAMENTE no lugar de `rerank`
abaixo, com a mesma assinatura `(query, chunks, *, top_n) -> list[...]`: trocar
o scorer não toca em `retrieve_and_rerank` nem nos Passos 1-2. É o ponto de troca.

Decisão de erro: este módulo PROPAGA exceções (igual retriever.py/hyde.py). A
tradução para `state["errors"]` é do nó RAG do grafo (Semana 6).
"""

import logging
from dataclasses import replace
from functools import lru_cache
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from finsight.db.session import settings
from finsight.retrieval.hyde import hyde_retrieve
from finsight.retrieval.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

# Over-fetch: quantos candidatos a busca vetorial traz ANTES do re-rank. 20 é o
# ponto clássico — grande o bastante para o chunk decisivo quase sempre estar no
# conjunto, pequeno o bastante para o LLM ranquear num prompt só sem estourar
# contexto/custo.
DEFAULT_FETCH_K = 20
# Quantos sobrevivem ao re-rank e vão para o contexto do LLM. Casa com o
# DEFAULT_TOP_K=5 do retriever baseline.
DEFAULT_TOP_N = 5

# Teto de caracteres por candidato enviado ao juiz. Trade-off: cortar demais pode
# remover justamente o trecho relevante (pior ranking); cortar de menos infla
# tokens/custo/latência com 20 candidatos. ~1200 chars (~300 tokens) cobre um
# chunk típico de relatório inteiro na maioria dos casos.
_MAX_CHARS_PER_CANDIDATE = 1200

# Escala da nota de relevância que pedimos ao LLM. 0..10 é intuitivo para o modelo
# e fácil de normalizar para [0, 1] (basta dividir por 10).
_RELEVANCE_SCALE = 10.0

_RERANK_SYSTEM_PROMPT = (
    "Você é um juiz de relevância para recuperação de trechos de relatórios "
    "financeiros. Receberá uma PERGUNTA e uma lista de TRECHOS candidatos, cada "
    "um com um índice numérico. Avalie cada trecho pela capacidade de RESPONDER "
    "diretamente à pergunta — premie dados concretos (métricas, períodos, "
    "rubricas) que atendam ao que foi perguntado e penalize trechos apenas "
    "tangenciais ou genéricos. Atribua a cada trecho uma relevância de 0 a 10 "
    "(10 = responde plenamente). Devolva TODOS os índices recebidos, ordenados "
    "do mais relevante para o menos relevante."
)


class RankedCandidate(BaseModel):
    """Uma nota de relevância para um candidato, referenciado pelo índice enviado."""

    index: int = Field(description="Índice (0-based) do trecho na lista enviada.")
    relevance: float = Field(
        description="Relevância de 0 a 10; 10 = responde plenamente à pergunta."
    )


class RerankResult(BaseModel):
    """
    Saída estruturada do juiz: idealmente TODOS os candidatos, ordenados.

    Usar structured output (em vez de parsear texto) é a convenção do projeto: o
    langchain injeta este schema no modelo e devolve o objeto já validado.
    """

    ranking: list[RankedCandidate] = Field(
        description="Candidatos ordenados do mais relevante ao menos relevante."
    )


@lru_cache(maxsize=1)
def _get_ranking_client() -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
    """
    Client de julgamento, já embrulhado com structured output. Ponto de mock.

    Diferença crucial para o HyDE: aqui o ponto de mock é o RUNNABLE JÁ
    EMBRULHADO por `.with_structured_output`, não o ChatOpenAI cru — é nele que o
    teste injeta `.ainvoke()` devolvendo um RerankResult determinístico.

    temperature=0.0: re-ranking é tarefa de JULGAMENTO, queremos determinismo e
    consistência — o oposto do HyDE (0.7), que busca diversidade para a média.
    """
    client = ChatOpenAI(
        api_key=SecretStr(settings.openai_api_key),
        model=settings.active_llm_model,
        temperature=0.0,
        max_retries=4,
        timeout=60.0,
    )
    return client.with_structured_output(RerankResult)


def _format_candidates(chunks: list[RetrievedChunk]) -> str:
    """
    Monta o bloco numerado de candidatos para o prompt.

    O índice impresso é a POSIÇÃO na lista (0-based), e é por ele que o LLM se
    refere a cada trecho na resposta — não pelo document_id. Manter o
    "endereçamento" local à chamada é o que torna o pós-processamento robusto:
    validamos os índices contra `len(chunks)`, sem depender de IDs do banco.
    """
    blocks = []
    for i, chunk in enumerate(chunks):
        # Trunca para conter custo/latência com fetch_k candidatos. Ver
        # _MAX_CHARS_PER_CANDIDATE para o trade-off.
        content = chunk.content[:_MAX_CHARS_PER_CANDIDATE]
        blocks.append(f"[{i}] {content}")
    return "\n\n".join(blocks)


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> list[RetrievedChunk]:
    """
    Reordena candidatos por relevância (LLM listwise) e devolve os `top_n`.

    Primitiva do re-ranking, agnóstica à ORIGEM dos candidatos (busca crua, HyDE,
    tanto faz) — espelha como `search_by_embedding` é agnóstica à origem do vetor.
    É o núcleo testável: o teste injeta `chunks` fabricados + mock do LLM e
    verifica a reordenação, sem tocar no banco.

    Pós-processamento defensivo (LLM não é confiável com índices):
      1. ignora índices fora de [0, len(chunks)) (alucinação) e DEDUPLICA;
      2. faz APPEND dos candidatos omitidos pelo LLM, no fim, na ordem original
         (cosine) — nunca PERDEMOS um chunk só porque o juiz esqueceu dele;
      3. corta em top_n.
    Com isso o re-rank só pode MELHORAR ou EMPATAR com o baseline vetorial, nunca
    destruir o recall que a busca já garantiu.

    Score resultante: `score` passa a ser a relevância do juiz normalizada para
    [0, 1]; o cosine original é preservado em `metadata["vector_score"]`. Assim o
    sinal mais fiel (pós-rerank) vira o score, mas dá para comparar os dois e
    medir o ganho da técnica.
    """
    # Guards: sem candidatos não há o que ranquear; com 0 ou 1 candidato, ou query
    # vazia, uma chamada de LLM seria desperdício — devolvemos o que já temos.
    if not chunks or len(chunks) == 1 or not query.strip():
        return chunks[:top_n]

    messages = [
        SystemMessage(content=_RERANK_SYSTEM_PROMPT),
        HumanMessage(content=f"PERGUNTA:\n{query}\n\nTRECHOS:\n{_format_candidates(chunks)}"),
    ]

    client = _get_ranking_client()
    result = await client.ainvoke(messages)
    # with_structured_output garante o tipo; o cast explícito é só para o mypy,
    # que vê o retorno genérico do Runnable.
    assert isinstance(result, RerankResult)

    # Passo 1 — consumir o ranking do LLM, validando índices e deduplicando.
    # `relevance_by_index` guarda a nota só dos candidatos que o LLM de fato
    # pontuou. ORDENAMOS pela `relevance` (decrescente), não pela ordem do array
    # devolvido: a nota é a fonte de verdade, e ordenar por ela (a) não depende
    # de o modelo acertar a ordem do array além das notas e (b) garante a mesma
    # invariante de retriever/HyDE — scores de saída em ordem decrescente.
    relevance_by_index: dict[int, float] = {}
    for item in result.ranking:
        if 0 <= item.index < len(chunks) and item.index not in relevance_by_index:
            relevance_by_index[item.index] = item.relevance

    # `sorted` é estável: empates de relevância preservam a ordem em que o LLM os
    # citou (já inserida na dict acima).
    ordered_indices: list[int] = sorted(
        relevance_by_index, key=lambda i: relevance_by_index[i], reverse=True
    )
    seen = set(ordered_indices)

    # Passo 2 — APPEND dos omitidos, na ordem original (cosine). Eles ficam atrás
    # dos pontuados, mas continuam disponíveis caso o top_n alcance até eles.
    for i in range(len(chunks)):
        if i not in seen:
            ordered_indices.append(i)

    if len(seen) < len(chunks):
        logger.debug(
            "Re-rank: LLM pontuou %d de %d candidatos; %d anexados na ordem original.",
            len(seen),
            len(chunks),
            len(chunks) - len(seen),
        )

    # Passo 3 — reconstruir os chunks na nova ordem, reescrevendo o score.
    reranked: list[RetrievedChunk] = []
    for i in ordered_indices:
        chunk = chunks[i]
        if i in relevance_by_index:
            # Normaliza 0..10 -> [0, 1] e faz clamp (o LLM pode extrapolar a escala).
            new_score = max(0.0, min(1.0, relevance_by_index[i] / _RELEVANCE_SCALE))
        else:
            # Omitido pelo juiz: sem nota de relevância, mantemos o cosine como
            # score para não inventar um número. Já está no fim da fila de qualquer modo.
            new_score = chunk.score
        # replace() cria uma cópia (dataclass) em vez de mutar o original — função
        # pura, sem efeito colateral em quem nos passou a lista. Preserva o cosine
        # em metadata["vector_score"] para comparação posterior.
        reranked.append(
            replace(
                chunk,
                score=new_score,
                metadata={**chunk.metadata, "vector_score": chunk.score},
            )
        )

    return reranked[:top_n]


async def retrieve_and_rerank(
    query: str,
    *,
    ticker: str | None = None,
    fetch_k: int = DEFAULT_FETCH_K,
    top_n: int = DEFAULT_TOP_N,
    use_hyde: bool = False,
    session: AsyncSession | None = None,
) -> list[RetrievedChunk]:
    """
    Pipeline completo: over-fetch (`fetch_k`) -> rerank -> `top_n`.

    Conveniência que compõe sobre os Passos 1-2 — espelha `retrieve`/`hyde_retrieve`.
    `use_hyde=True` faz o over-fetch via `hyde_retrieve` em vez da busca crua:
    as duas técnicas da Semana 3 EMPILHADAS — recall esperto (HyDE) + ordenação
    fina (re-rank).

    Args:
        query: pergunta do usuário.
        ticker: restringe a busca a um ativo (repassado ao retriever).
        fetch_k: quantos candidatos buscar antes de reordenar (over-fetch).
        top_n: quantos devolver após o re-rank.
        use_hyde: usa HyDE no estágio de busca se True; busca vetorial crua se False.
        session: sessão a reusar (emprestada); se None, o retriever abre a própria.
    """
    if not query.strip():
        return []

    # Over-fetch: buscamos fetch_k (grande), não top_n. É o que dá ao re-ranker
    # material para corrigir a ordem fina.
    if use_hyde:
        candidates = await hyde_retrieve(query, ticker=ticker, top_k=fetch_k, session=session)
    else:
        candidates = await retrieve(query, ticker=ticker, top_k=fetch_k, session=session)

    return await rerank(query, candidates, top_n=top_n)
