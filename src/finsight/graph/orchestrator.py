"""
Orchestrator — monta e compila o grafo LangGraph que costura os agentes.

Este módulo liga os nós num StateGraph executável, num LEQUE (fan-out -> fan-in):

                      START
                   /    |    \
          financial  research  rag      (rodam em PARALELO, mesmo superstep)
                   \\    |    /
                       risk             (FAN-IN: só roda após os TRÊS terminarem)
                        |
                       END

>>> Por que diamante e como o LangGraph o executa <<<
- Arestas saindo de START = FAN-OUT: os três nós disparam concorrentemente.
  (Por isso financial/research usam asyncio.to_thread nos seus IO bloqueantes.)
- Arestas chegando em `risk` = FAN-IN: o LangGraph espera TODOS os ramos
  completarem antes de rodar `risk` uma única vez, já com o state mesclado. Modelo
  de superstep — não precisamos de espera manual.
- Erros dos ramos paralelos se ACUMULAM: `AgentState.errors` tem reducer `add`, que
  concatena as escritas concorrentes em vez de uma sobrescrever a outra. Quando
  `risk` roda, `state["errors"]` já tem os erros dos três ramos.
"""

import logging
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from finsight.agents.financial import financial_node
from finsight.agents.rag import rag_node
from finsight.agents.research import research_node
from finsight.agents.risk import risk_node
from finsight.graph.state import AgentState

logger = logging.getLogger(__name__)

# Nomes dos nós em um só lugar — evita strings mágicas espalhadas nas arestas e nos
# testes. Os ramos paralelos do fan-out e o nó de síntese (fan-in).
_NODE_FINANCIAL = "financial"
_NODE_RESEARCH = "research"
_NODE_RAG = "rag"
_NODE_RISK = "risk"


def build_orchestrator() -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """
    Monta e compila o grafo do diamante.

    Sem checkpointer por ora: a persistência de state (Redis/checkpointing) é da
    Semana 7. Compilar é barato e sem efeito de rede — pode ser chamado no startup
    da app e reutilizado por todas as requisições.
    """
    builder: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)

    # Registra os nós. add_node guarda a REFERÊNCIA da função; o nó resolve seus
    # pontos de rede (_fetch_prices, _search_news, _get_chat_client) em tempo de
    # chamada — é o que permite os testes mockarem sem rede.
    builder.add_node(_NODE_FINANCIAL, financial_node)
    builder.add_node(_NODE_RESEARCH, research_node)
    builder.add_node(_NODE_RAG, rag_node)
    builder.add_node(_NODE_RISK, risk_node)

    # FAN-OUT: START -> cada ramo (disparam em paralelo, mesmo superstep).
    builder.add_edge(START, _NODE_FINANCIAL)
    builder.add_edge(START, _NODE_RESEARCH)
    builder.add_edge(START, _NODE_RAG)

    # FAN-IN: todos -> risk. risk só roda quando os três ramos terminam.
    builder.add_edge(_NODE_FINANCIAL, _NODE_RISK)
    builder.add_edge(_NODE_RESEARCH, _NODE_RISK)
    builder.add_edge(_NODE_RAG, _NODE_RISK)

    # risk -> END: fim da execução.
    builder.add_edge(_NODE_RISK, END)

    compiled = builder.compile()
    logger.debug("orchestrator: grafo compilado (financial+research+rag -> risk)")
    return compiled


def build_initial_state(query: str, ticker: str, asset_type: str = "stock") -> AgentState:
    """
    Constrói o AgentState de entrada — único lugar que sabe a forma do state inicial.

    Popula os inputs, gera um `execution_id` (uuid4, para correlacionar logs/traces
    da Semana 7), zera os outputs (None até cada nó completar) e inicializa `errors`
    como lista vazia (o reducer `add` exige que a chave exista para concatenar).
    """
    return AgentState(
        query=query,
        ticker=ticker,
        asset_type=asset_type,
        research=None,
        financial=None,
        rag=None,
        final_answer=None,
        execution_id=str(uuid4()),
        cached=False,
        errors=[],
    )


async def run_analysis(query: str, ticker: str, asset_type: str = "stock") -> AgentState:
    """
    Conveniência de ponta a ponta: monta o state inicial e executa o grafo.

    É o que a API SSE (Semana 6) vai chamar. `ainvoke` roda o grafo até END e devolve
    o state final mesclado (com financial/research/final_answer preenchidos). Compila
    o grafo a cada chamada por simplicidade; a app pode cachear `build_orchestrator()`.
    """
    graph = build_orchestrator()
    initial = build_initial_state(query, ticker, asset_type)
    result = await graph.ainvoke(initial)
    return result  # type: ignore[return-value]
