"""
AgentState e modelos de output de cada agente.

Este arquivo é o contrato central do sistema: todo nó do grafo LangGraph
lê e escreve nestes tipos. Alterar campos aqui tem impacto em cascata.
"""

from operator import add
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Modelos de output — um por agente especializado
# Usam BaseModel (não TypedDict) porque são validados via structured output
# do LLM (model.with_structured_output(ResearchOutput)) e precisam de
# serialização JSON para o cache Redis.
# ---------------------------------------------------------------------------


class ResearchOutput(BaseModel):
    """Output do Research Agent (Tavily + análise de sentimento)."""

    # strict=True: o LLM deve retornar os tipos exatos — sem coerção silenciosa.
    # Se o modelo retornar sentiment="0.8" (string) em vez de float, queremos
    # um ValidationError explícito, não um bug silencioso.
    model_config = ConfigDict(strict=True)

    summary: str = Field(description="Resumo das notícias e eventos recentes do ativo")
    sentiment: str = Field(
        description="Sentimento do mercado em relação ao ativo",
        pattern="^(bullish|bearish|neutral)$",  # enum via regex — evita typos do LLM
    )
    key_events: list[str] = Field(
        description="Lista de eventos relevantes encontrados nas notícias"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confiança do agente na análise (0-1)"
    )
    sources: list[str] = Field(
        default_factory=list, description="URLs das fontes consultadas"
    )


class FinancialOutput(BaseModel):
    """Output do Financial Agent (yfinance + cálculos quantitativos)."""

    model_config = ConfigDict(strict=True)

    # Todos os campos são opcionais porque yfinance pode falhar para ativos
    # com histórico curto (ex: FIIs recém-listados) — preferimos None a raise.
    sharpe_ratio: float | None = Field(
        default=None, description="Índice de Sharpe anualizado (janela 1 ano)"
    )
    var_95: float | None = Field(
        default=None, description="Value at Risk a 95% de confiança (diário)"
    )
    cumulative_return_1y: float | None = Field(
        default=None, description="Retorno acumulado dos últimos 252 dias úteis"
    )
    volatility_annualized: float | None = Field(
        default=None, description="Volatilidade anualizada (desvio padrão * sqrt(252))"
    )
    current_price: float | None = Field(
        default=None, description="Último preço de fechamento disponível"
    )
    data_source: str = Field(default="Yahoo Finance")


class RAGOutput(BaseModel):
    """Output do RAG Agent (busca por similaridade no pgvector)."""

    model_config = ConfigDict(strict=True)

    chunks: list[str] = Field(
        default_factory=list,
        description="Trechos de documentos recuperados por similaridade",
    )
    sources: list[str] = Field(
        default_factory=list, description="Títulos/IDs dos documentos de origem"
    )
    relevance_scores: list[float] = Field(
        default_factory=list,
        description="Score de similaridade cosine para cada chunk (0-1)",
    )


class FinalAnswer(BaseModel):
    """Output do Risk Agent — resposta final sintetizada para o usuário."""

    model_config = ConfigDict(strict=True)

    analysis: str = Field(
        description="Análise completa em linguagem natural, integrando dados dos 3 agentes"
    )
    key_points: list[str] = Field(
        description="Pontos principais da análise em formato de bullet points"
    )
    risk_factors: list[str] = Field(
        description="Fatores de risco identificados para o ativo"
    )
    # disclaimer fixo — não gerado pelo LLM, garantido pelo schema
    disclaimer: str = Field(
        default="Esta análise não constitui recomendação de investimento."
    )


# ---------------------------------------------------------------------------
# AgentState — estado central do grafo LangGraph
#
# TypedDict (não BaseModel) porque o LangGraph serializa o estado internamente
# como dict puro para o checkpointer. BaseModel teria overhead de conversão
# a cada transição de nó.
#
# Campos com Annotated + reducer: quando dois nós paralelos (fan-out) escrevem
# na mesma chave, o LangGraph precisa saber como mesclar os valores.
# `add` (operator.add) sobre listas faz concatenação — erros de múltiplos
# agentes são acumulados, não sobrescritos.
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    # --- Inputs (definidos na entrada da query) ---
    query: str
    ticker: str       # ex: "PETR4", "VALE3", "KNRI11"
    asset_type: str   # ex: "stock", "fii", "etf"

    # --- Outputs dos agentes (None até o agente respectivo completar) ---
    # Opcionais explícitos: o orquestrador precisa checar se o agente já rodou
    # antes de tentar acessar o output (evita KeyError em conditional edges)
    research: ResearchOutput | None
    financial: FinancialOutput | None
    rag: RAGOutput | None
    final_answer: FinalAnswer | None

    # --- Controle de execução ---
    execution_id: str  # UUID gerado na entrada — usado para correlacionar logs/traces
    cached: bool       # True se a resposta veio do Redis; False se foi computada agora

    # Annotated com reducer `add`: lista acumulada de erros de todos os nós.
    # Cada nó captura suas exceções e faz `return {"errors": ["mensagem"]}`.
    # O LangGraph concatena automaticamente — nunca sobrescreve erros anteriores.
    errors: Annotated[list[str], add]
