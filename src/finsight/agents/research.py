"""
Research Agent — notícias recentes + análise de sentimento do ativo.

É o complemento do Financial Agent (Passo 1): aquele respondia "o que os NÚMEROS
dizem?" (Sharpe, VaR, retorno) de forma determinística; este responde "o que o
MERCADO está DIZENDO?" — manchetes recentes resumidas e classificadas em
bullish/bearish/neutral. É o primeiro nó com LLM de verdade no grafo. Ambos rodam
em paralelo no fan-out (Passo 3); por isso `AgentState.errors` tem reducer `add`:
falhas dos dois se acumulam sem sobrescrever.

>>> Duas camadas (mesmo padrão do financial.py / retrieval), mas DOIS pontos de rede <<<

    _search_news(ticker, query) -> list[NewsItem]    # IO #1: Tavily   (ponto de mock)
    analyze_sentiment(query, news) -> ResearchOutput  # IO #2: LLM      (ponto de mock)
    research_node(state) -> dict[str, Any]            # NÓ: costura, captura erro

`_search_news` espelha o `_fetch_prices`: chamada SÍNCRONA e bloqueante (a SDK do
Tavily faz HTTP), então o nó a invoca via `asyncio.to_thread`. `analyze_sentiment`
é o núcleo de raciocínio — recebe as notícias JÁ buscadas (agnóstico à origem,
igual `generate_answer` recebia `list[str]`) e devolve o output estruturado.

>>> Decisão não-óbvia: o LLM NÃO inventa as `sources` <<<
O LLM julga (resume, classifica sentimento, extrai eventos), mas URLs saídas de um
LLM são candidatas a alucinação. Então o structured output usa um schema PRIVADO
`_SentimentVerdict` SEM `sources`; as URLs reais são coladas pelo código a partir
dos `NewsItem` que o Tavily devolveu. Verdade factual = dado real; julgamento = LLM.
(Mesmo princípio do reranker: a nota vinha do juiz, mas a ordem-verdade era imposta
pelo código.)

>>> Degradação graciosa (regra de erro do projeto) <<<
Sem notícias -> guard determinístico devolve um output neutro SEM chamar o LLM.
Tavily ou LLM caem -> o nó captura, devolve output neutro + `{"errors": [msg]}`,
NUNCA propaga. O grafo segue com a síntese parcial.
"""

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from tavily import TavilyClient

from finsight.db.session import settings
from finsight.graph.state import AgentState, ResearchOutput

logger = logging.getLogger(__name__)

# Quantas manchetes pedir ao Tavily. Poucas o bastante para caber no prompt sem
# estourar custo/contexto; muitas o bastante para o sentimento não depender de uma
# notícia só. Ajustável; 5 é o equilíbrio para o golden set.
_MAX_NEWS_RESULTS = 5

# Sentimento padrão quando não há sinal (sem notícias / erro): neutro, confiança 0.
# "neutral" casa com a regex do ResearchOutput; confiança 0 sinaliza "sem base".
_NEUTRAL_SENTIMENT = "neutral"

# Termos que ANCORAM a busca no domínio financeiro. Sem eles, o código do ticker
# (ex: PETR4) + a pergunta em linguagem natural faziam o Tavily devolver notícias
# genéricas/irrelevantes (esporte, etc.). Mantêm a busca em "mercado/ações".
_NEWS_QUERY_ANCHOR = "ações mercado financeiro"

# Janela de recência da busca (dias). Sentimento de mercado é sobre o AGORA — limitar
# a manchetes recentes evita notícia velha pesar no veredito.
_NEWS_RECENCY_DAYS = 30


# ---------------------------------------------------------------------------
# Notícia normalizada — desacopla o resto do módulo do formato cru do Tavily
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsItem:
    """
    Uma manchete normalizada. O Tavily devolve dicts com várias chaves; reduzimos
    ao que importa (título, URL, trecho) para o prompt e para as `sources`. frozen:
    notícia é um fato imutável depois de buscada.
    """

    title: str
    url: str
    content: str


# ---------------------------------------------------------------------------
# Schema PRIVADO do structured output — note a ausência de `sources`
# ---------------------------------------------------------------------------


class _SentimentVerdict(BaseModel):
    """
    O que pedimos ao LLM produzir a partir das notícias. DELIBERADAMENTE sem
    `sources`: as URLs reais vêm dos NewsItem (código), não do modelo.

    `sentiment` é `Literal` (não regex como no ResearchOutput público): trava o
    enum na própria geração — o LLM só pode emitir um dos três valores, em vez de
    torcermos para a saída bater a regex depois.
    """

    summary: str = Field(description="Resumo conciso (2-4 frases) das notícias recentes do ativo.")
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        description="Sentimento agregado do mercado: bullish (otimista), bearish "
        "(pessimista) ou neutral (misto/sem direção clara)."
    )
    key_events: list[str] = Field(
        description="Eventos concretos extraídos das notícias (ex: resultado "
        "trimestral, fato relevante, mudança regulatória). Lista vazia se nenhum."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança na análise (0-1): alta quando as notícias são "
        "abundantes e convergentes; baixa quando escassas ou contraditórias.",
    )


# O prompt fixa o PAPEL e as REGRAS. Crucial: ancorar SÓ nas notícias fornecidas
# (anti-alucinação, como o generator do RAG) e calibrar a confiança pela quantidade
# e convergência do material — para o sentimento ser honesto, não confiante por
# inércia. Em PT-BR, coerente com o resto do projeto.
_SENTIMENT_SYSTEM_PROMPT = (
    "Você é um analista financeiro sênior especializado em análise de sentimento de "
    "mercado. Você recebe a lista NUMERADA de notícias recentes sobre um ativo e deve "
    "produzir uma análise estruturada. Regras: (1) baseie-se EXCLUSIVAMENTE nas notícias "
    "fornecidas — não use conhecimento externo nem invente fatos; (2) classifique o "
    "sentimento agregado como bullish, bearish ou neutral; (3) extraia apenas eventos "
    "concretos que aparecem nas notícias; (4) calibre a confiança pela quantidade e pela "
    "convergência das notícias — poucas ou contraditórias significam confiança baixa."
)


# ---------------------------------------------------------------------------
# Clients (pontos de mock) — um por serviço externo
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_tavily_client() -> TavilyClient:
    """Client do Tavily, criado uma vez (lru_cache). Ponto de mock de busca."""
    return TavilyClient(api_key=settings.tavily_api_key)


@lru_cache(maxsize=1)
def _get_chat_client() -> ChatOpenAI:
    """
    Client de chat CRU (sem structured output), criado uma vez. Ponto de mock do LLM.

    temperature=0.0: análise de sentimento é julgamento — mesma entrada, mesma nota
    (reprodutibilidade), igual ao reranker/metrics. Contrasta com o HyDE (0.7), que
    quer diversidade. O teste mocka este client e faz `.with_structured_output`
    devolver um runnable determinístico.
    """
    return ChatOpenAI(
        api_key=SecretStr(settings.openai_api_key),
        model=settings.active_llm_model,
        temperature=0.0,
        max_retries=4,
        timeout=60.0,
    )


# ---------------------------------------------------------------------------
# IO #1 — busca de notícias (Tavily)
# ---------------------------------------------------------------------------


def _build_news_query(ticker: str, query: str) -> str:
    """
    Monta a query de busca de notícias a partir do ticker e da pergunta do usuário.

    Por que não `f"{ticker} {query}"` cru (o que se usava antes): o código do ticker
    raramente aparece em manchetes e a pergunta em linguagem natural ("quais os
    destaques...") dilui a busca com palavras interrogativas — o Tavily devolvia
    notícias fora do domínio (esporte, etc.). Ancoramos em termos financeiros
    explícitos para manter a busca em mercado/ações, com o ticker à frente e a
    intenção do usuário em seguida. Pura e sem rede -> testável isoladamente.
    """
    return f"{ticker} {_NEWS_QUERY_ANCHOR} {query}".strip()


def _search_news(ticker: str, query: str) -> list[NewsItem]:
    """
    Busca notícias recentes do ativo via Tavily. SÍNCRONO e bloqueante (HTTP) — por
    isso o nó o chama via `asyncio.to_thread`. Único ponto de rede de busca e, logo,
    o ponto de mock.

    topic="news": pede ao Tavily o índice de notícias (não a web geral). A query vem
    do `_build_news_query` (ancorada em termos financeiros) e `days` limita à janela
    recente. Normalizamos o dict cru em NewsItem; resultados sem URL ou título são
    descartados (não servem nem de fonte nem de evidência).
    """
    client = _get_tavily_client()
    response = client.search(
        query=_build_news_query(ticker, query),
        topic="news",
        max_results=_MAX_NEWS_RESULTS,
        days=_NEWS_RECENCY_DAYS,
    )
    results = response.get("results", []) if isinstance(response, dict) else []

    items: list[NewsItem] = []
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        if not url or not title:
            continue
        items.append(NewsItem(title=title, url=url, content=r.get("content", "")))
    logger.debug("research: Tavily devolveu %d notícia(s) para %s", len(items), ticker)
    return items


# ---------------------------------------------------------------------------
# IO #2 — análise de sentimento (LLM)
# ---------------------------------------------------------------------------


def _format_news(news: list[NewsItem]) -> str:
    """Numera as notícias no prompt (título + trecho). Mesma convenção do reranker."""
    blocks = []
    for i, item in enumerate(news):
        blocks.append(f"[{i}] {item.title}\n{item.content}")
    return "\n\n".join(blocks)


def _empty_research(*, summary: str) -> ResearchOutput:
    """
    Output neutro de degradação graciosa: sem direção, confiança zero, sem fontes.
    Centraliza a forma do "sem sinal" usada pelo guard de notícias-vazias e pelo
    tratamento de erro do nó.
    """
    return ResearchOutput(
        summary=summary,
        sentiment=_NEUTRAL_SENTIMENT,
        key_events=[],
        confidence=0.0,
        sources=[],
    )


async def analyze_sentiment(query: str, news: list[NewsItem]) -> ResearchOutput:
    """
    Núcleo de raciocínio: das notícias buscadas -> ResearchOutput estruturado.

    Agnóstico à origem das notícias (recebe `list[NewsItem]`, não chama o Tavily) —
    espelha `compute_metrics`/`generate_answer`. O LLM produz o `_SentimentVerdict`
    (resumo/sentimento/eventos/confiança); as `sources` são coladas aqui a partir
    das URLs REAIS dos NewsItem, não do modelo.

    Guard sem LLM: sem notícias -> output neutro determinístico (recall zero do lado
    do Research; o diagnóstico certo é "sem base", não uma análise inventada).
    """
    if not news:
        return _empty_research(summary="Nenhuma notícia recente encontrada para o ativo.")

    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=_SENTIMENT_SYSTEM_PROMPT),
        HumanMessage(content=f"Pergunta do usuário: {query}\n\nNotícias:\n{_format_news(news)}"),
    ]

    client = _get_chat_client()
    # with_structured_output força o LLM a devolver o schema; cada nó com LLM faz o
    # embrulho na hora (client cru é o ponto de mock — vários nós, vários schemas).
    structured: Runnable[LanguageModelInput, dict[str, Any] | BaseModel] = (
        client.with_structured_output(_SentimentVerdict)
    )
    verdict = await structured.ainvoke(messages)
    # O Runnable é genérico para o mypy; o assert estreita o tipo (mesma defesa do
    # reranker/metrics) e garante o contrato em runtime.
    assert isinstance(verdict, _SentimentVerdict)

    # Junta o JULGAMENTO do LLM com os DADOS reais: as fontes são as URLs do Tavily.
    return ResearchOutput(
        summary=verdict.summary,
        sentiment=verdict.sentiment,
        key_events=verdict.key_events,
        confidence=verdict.confidence,
        sources=[item.url for item in news],
    )


# ---------------------------------------------------------------------------
# NÓ do grafo
# ---------------------------------------------------------------------------


async def research_node(state: AgentState) -> dict[str, Any]:
    """
    Nó do LangGraph: busca notícias do `ticker`/`query` do state e analisa o sentimento.

    Devolve o patch parcial `{"research": ResearchOutput}`. Em falha (Tavily fora do
    ar, LLM indisponível) captura a exceção e devolve, além do output neutro,
    `{"errors": [...]}` — degradação graciosa, nunca propaga. Os dois pontos de rede
    estão dentro do mesmo try: qualquer um que falhe vira erro registrado, e a
    síntese final segue com o que os outros agentes trouxeram.
    """
    ticker = state["ticker"]
    query = state["query"]
    try:
        # to_thread: a busca do Tavily é bloqueante; roda num worker thread para não
        # travar o event loop dos outros nós do fan-out (ex: Financial).
        news = await asyncio.to_thread(_search_news, ticker, query)
        research = await analyze_sentiment(query, news)
    except Exception as exc:  # fronteira do nó: captura tudo, nunca propaga
        logger.warning("research_node: falha ao analisar %s: %s", ticker, exc)
        return {
            "research": _empty_research(summary=f"Falha ao buscar/analisar notícias de {ticker}."),
            "errors": [f"research: falha ao analisar {ticker}: {exc}"],
        }

    logger.debug(
        "research_node: %s -> sentiment=%s confidence=%.2f sources=%d",
        ticker,
        research.sentiment,
        research.confidence,
        len(research.sources),
    )
    return {"research": research}
