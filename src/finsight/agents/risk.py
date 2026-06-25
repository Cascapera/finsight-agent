"""
Risk Agent — síntese final que integra os outputs dos agentes especializados.

É o nó de FAN-IN do grafo (Passo 3): roda DEPOIS de financial_node e research_node
(e, na Semana 6, do RAG Agent), quando o state já tem os outputs dos ramos
paralelos mesclados. Sua função é transformar dados estruturados de fontes
heterogêneas (métricas quant + sentimento de notícias + chunks de relatórios) numa
análise em linguagem natural — o `FinalAnswer` que volta para o usuário.

>>> Duas camadas (mesmo padrão dos outros nós) <<<

    synthesize(state) -> FinalAnswer    # núcleo de raciocínio: state -> análise
    risk_node(state)  -> dict           # NÓ: chama synthesize, captura erro

`synthesize` lê o que estiver DISPONÍVEL no state (financial/research/rag podem ser
None se um ramo não rodou ou falhou) e produz uma análise honesta mesmo com dados
parciais — degradação graciosa de ponta a ponta.

>>> Decisão não-óbvia: o LLM NÃO gera o `disclaimer` <<<
Texto regulatório não pode depender da boa vontade do modelo. O structured output
usa um schema PRIVADO `_RiskVerdict` SEM disclaimer; o `FinalAnswer` final aplica o
disclaimer FIXO (default do schema). Mesmo princípio das `sources` do Research: o
LLM julga, o código garante o que é factual/obrigatório.
"""

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from finsight.db.session import settings
from finsight.graph.state import (
    AgentState,
    FinalAnswer,
    FinancialOutput,
    RAGOutput,
    ResearchOutput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema PRIVADO do structured output — note a ausência de `disclaimer`
# ---------------------------------------------------------------------------


class _RiskVerdict(BaseModel):
    """
    O que pedimos ao LLM na síntese. DELIBERADAMENTE sem `disclaimer`: ele é fixo e
    obrigatório, garantido pelo default do FinalAnswer (código), não pelo modelo.
    """

    analysis: str = Field(
        description="Análise integrada em linguagem natural (3-6 frases), cruzando "
        "métricas quantitativas, sentimento de notícias e trechos de relatórios."
    )
    key_points: list[str] = Field(
        description="Pontos principais da análise, em bullets concisos."
    )
    risk_factors: list[str] = Field(
        description="Fatores de risco concretos identificados a partir dos dados."
    )


# O prompt fixa o papel de SINTETIZADOR e a regra de ancoragem: usar SÓ os dados
# fornecidos pelos agentes, sem inventar números nem notícias. É o mesmo princípio
# anti-alucinação do generator/research, agora sobre dados já estruturados.
_RISK_SYSTEM_PROMPT = (
    "Você é um analista de risco sênior. Você recebe os resultados de três análises "
    "sobre um ativo: métricas quantitativas (risco/retorno), sentimento de notícias e "
    "trechos de relatórios. Sua tarefa é produzir uma síntese integrada para o "
    "investidor. Regras: (1) baseie-se EXCLUSIVAMENTE nos dados fornecidos — não invente "
    "métricas, notícias ou fatos; (2) cruze as fontes (ex: 'apesar do Sharpe alto, o "
    "sentimento bearish e o VaR elevado sugerem cautela'); (3) seja explícito sobre "
    "lacunas quando algum dado estiver ausente; (4) não inclua disclaimer — ele é "
    "adicionado automaticamente."
)


@lru_cache(maxsize=1)
def _get_chat_client() -> ChatOpenAI:
    """
    Client de chat CRU, criado uma vez (lru_cache). Ponto de mock do LLM.

    temperature=0.0: síntese é raciocínio determinístico sobre dados dados — mesma
    entrada, mesma análise. Igual ao reranker/metrics/research.
    """
    return ChatOpenAI(
        api_key=SecretStr(settings.openai_api_key),
        model=settings.active_llm_model,
        temperature=0.0,
        max_retries=4,
        timeout=60.0,
    )


# ---------------------------------------------------------------------------
# Serialização dos outputs dos agentes para o prompt
# ---------------------------------------------------------------------------


def _format_financial(fin: FinancialOutput | None) -> str:
    """Serializa as métricas quant; sinaliza ausência explicitamente."""
    if fin is None:
        return "MÉTRICAS QUANTITATIVAS: indisponíveis."
    return (
        "MÉTRICAS QUANTITATIVAS:\n"
        f"- Preço atual: {fin.current_price}\n"
        f"- Sharpe (1a): {fin.sharpe_ratio}\n"
        f"- VaR 95% (diário): {fin.var_95}\n"
        f"- Retorno acumulado (1a): {fin.cumulative_return_1y}\n"
        f"- Volatilidade anualizada: {fin.volatility_annualized}"
    )


def _format_research(res: ResearchOutput | None) -> str:
    """Serializa o sentimento + eventos; sinaliza ausência explicitamente."""
    if res is None:
        return "ANÁLISE DE NOTÍCIAS: indisponível."
    events = "; ".join(res.key_events) if res.key_events else "(nenhum evento listado)"
    return (
        "ANÁLISE DE NOTÍCIAS:\n"
        f"- Sentimento: {res.sentiment} (confiança {res.confidence:.2f})\n"
        f"- Resumo: {res.summary}\n"
        f"- Eventos: {events}"
    )


def _format_rag(rag: RAGOutput | None) -> str:
    """
    Serializa os chunks de relatórios. Hoje sempre None (RAG Agent é Semana 6);
    já tratado para que plugar o ramo RAG depois não exija tocar nesta função.
    """
    if rag is None or not rag.chunks:
        return "TRECHOS DE RELATÓRIOS: indisponíveis."
    joined = "\n".join(f"- {c}" for c in rag.chunks)
    return f"TRECHOS DE RELATÓRIOS:\n{joined}"


def _fallback_answer(*, analysis: str) -> FinalAnswer:
    """
    FinalAnswer de degradação: análise mínima, sem pontos/riscos derivados. O
    disclaimer vem do default do schema (sempre presente). Usado pelo nó quando o
    LLM da síntese falha.
    """
    return FinalAnswer(analysis=analysis, key_points=[], risk_factors=[])


# ---------------------------------------------------------------------------
# Núcleo de raciocínio
# ---------------------------------------------------------------------------


async def synthesize(state: AgentState) -> FinalAnswer:
    """
    Cruza os outputs dos agentes (no state) numa análise integrada -> FinalAnswer.

    Lê `financial`/`research`/`rag` de forma defensiva (`.get`): qualquer um pode ser
    None. Serializa o que há, e o LLM produz o `_RiskVerdict`; o `disclaimer` do
    FinalAnswer é o fixo do schema, não do modelo.
    """
    financial = state.get("financial")
    research = state.get("research")
    rag = state.get("rag")

    context = "\n\n".join(
        [
            f"Pergunta do usuário: {state['query']}",
            f"Ativo: {state['ticker']} ({state.get('asset_type', 'desconhecido')})",
            _format_financial(financial),
            _format_research(research),
            _format_rag(rag),
        ]
    )

    client = _get_chat_client()
    structured: Runnable[LanguageModelInput, dict[str, Any] | BaseModel] = (
        client.with_structured_output(_RiskVerdict)
    )
    verdict = await structured.ainvoke(
        [
            SystemMessage(content=_RISK_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    )
    # Runnable é genérico para o mypy; assert estreita o tipo (mesma defesa dos
    # outros nós LLM) e garante o contrato em runtime.
    assert isinstance(verdict, _RiskVerdict)

    # disclaimer NÃO vem do verdict: é o default fixo do FinalAnswer (código).
    return FinalAnswer(
        analysis=verdict.analysis,
        key_points=verdict.key_points,
        risk_factors=verdict.risk_factors,
    )


# ---------------------------------------------------------------------------
# NÓ do grafo (fan-in)
# ---------------------------------------------------------------------------


async def risk_node(state: AgentState) -> dict[str, Any]:
    """
    Nó de síntese final: integra os outputs dos ramos paralelos -> {"final_answer": ...}.

    Roda no fan-in do diamante — quando chega aqui, `state["errors"]` já acumulou
    (via reducer `add`) os erros dos ramos paralelos. Captura sua própria exceção
    (LLM fora do ar) e devolve um FinalAnswer de fallback + erro, nunca propaga.
    """
    try:
        final_answer = await synthesize(state)
    except Exception as exc:  # fronteira do nó: captura tudo, nunca propaga
        logger.warning("risk_node: falha na síntese de %s: %s", state["ticker"], exc)
        return {
            "final_answer": _fallback_answer(
                analysis=f"Não foi possível gerar a síntese final para {state['ticker']}."
            ),
            "errors": [f"risk: falha na síntese de {state['ticker']}: {exc}"],
        }

    logger.debug(
        "risk_node: %s -> %d pontos, %d riscos",
        state["ticker"],
        len(final_answer.key_points),
        len(final_answer.risk_factors),
    )
    return {"final_answer": final_answer}
