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
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from finsight.agents.financial import financial_node
from finsight.agents.rag import rag_node
from finsight.agents.research import research_node
from finsight.agents.risk import risk_node
from finsight.graph.state import AgentState
from finsight.observability.metrics import NodeFn, instrument_node

logger = logging.getLogger(__name__)

# Nomes dos nós em um só lugar — evita strings mágicas espalhadas nas arestas e nos
# testes. Os ramos paralelos do fan-out e o nó de síntese (fan-in).
_NODE_FINANCIAL = "financial"
_NODE_RESEARCH = "research"
_NODE_RAG = "rag"
_NODE_RISK = "risk"


def _add_node(
    builder: "StateGraph[AgentState, Any, Any, Any]",
    name: str,
    fn: NodeFn,
) -> None:
    """
    Registra um nó já instrumentado com métricas.

    O `type: ignore[call-overload]` é fricção dos stubs do langgraph: os overloads de
    `add_node` casam com uma `async def` concreta, mas não com um `Callable` devolvido
    por outra função (instrument_node) via alias. Em runtime é a mesma corrotina;
    centralizar aqui isola o ignore num único ponto (mesmo espírito do ignore de
    `ainvoke` em run_analysis).
    """
    builder.add_node(name, instrument_node(name, fn))  # type: ignore[call-overload]


def build_orchestrator() -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """
    Monta e compila o grafo do diamante.

    Sem checkpointer por ora: a persistência de state (Redis/checkpointing) é da
    Semana 7. Compilar é barato e sem efeito de rede — pode ser chamado no startup
    da app e reutilizado por todas as requisições.
    """
    builder: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)

    # Registra os nós, cada um embrulhado por instrument_node (métricas Prometheus:
    # latência/runs/erros por nó). O nó resolve seus pontos de rede (_fetch_prices,
    # _search_news, _get_chat_client) em tempo de chamada — é o que permite os testes
    # mockarem sem rede. O wrapper é transparente ao mock: só envolve, a função
    # original segue resolvendo tudo. _add_node centraliza o cast (ver helper abaixo).
    _add_node(builder, _NODE_FINANCIAL, financial_node)
    _add_node(builder, _NODE_RESEARCH, research_node)
    _add_node(builder, _NODE_RAG, rag_node)
    _add_node(builder, _NODE_RISK, risk_node)

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

    `ainvoke` roda o grafo até END e devolve o state final mesclado (financial/
    research/rag/final_answer preenchidos). BLOQUEANTE: só retorna quando tudo acaba.
    Para progresso incremental (API SSE), use `run_analysis_stream`. Compila o grafo a
    cada chamada por simplicidade; a app pode cachear `build_orchestrator()`.
    """
    graph = build_orchestrator()
    initial = build_initial_state(query, ticker, asset_type)
    result = await graph.ainvoke(initial)
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Streaming — eventos por nó para a API SSE
# ---------------------------------------------------------------------------


class AnalysisEvent(BaseModel):
    """
    Evento de domínio emitido durante a execução do grafo. JSON-serializável.

    DELIBERADAMENTE agnóstico a HTTP/SSE: o orquestrador não conhece a camada de
    transporte. A API (Passo 3) traduz cada AnalysisEvent num evento SSE. Manter essa
    fronteira é o que torna o streaming testável sem subir um servidor.

    type="progress": um nó terminou (`node` = qual; `data` = patch serializado).
    type="complete": fim da execução (`data` = final_answer completo + erros acumulados).
    """

    type: str
    node: str | None = None
    data: dict[str, Any] = {}


def _serialize_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """
    Converte o patch de um nó em algo JSON-serializável.

    O patch pode trazer modelos Pydantic (FinancialOutput, ResearchOutput, ...) e/ou
    `errors: list[str]`. Despejamos os modelos com `model_dump()`; o resto passa direto.
    """
    out: dict[str, Any] = {}
    for key, value in patch.items():
        out[key] = value.model_dump() if isinstance(value, BaseModel) else value
    return out


async def run_analysis_stream(
    query: str, ticker: str, asset_type: str = "stock"
) -> AsyncIterator[AnalysisEvent]:
    """
    Executa o grafo emitindo um evento por nó concluído + um evento final.

    Usa `astream(stream_mode="updates")`: o LangGraph entrega `{nome_do_nó: patch}` a
    cada nó que termina (os ramos do fan-out chegam conforme completam). Para cada
    update emitimos um evento "progress"; ao fim, um "complete" com o final_answer e a
    lista ACUMULADA de erros — como `updates` traz só patches (não o state cheio),
    acumulamos final_answer (vem no patch do `risk`) e os erros durante o stream.
    """
    graph = build_orchestrator()
    initial = build_initial_state(query, ticker, asset_type)

    final_answer: dict[str, Any] | None = None
    errors: list[str] = []

    async for update in graph.astream(initial, stream_mode="updates"):
        # Um update pode, em tese, conter mais de um nó (mesmo superstep); iteramos.
        for node_name, patch in update.items():
            errors.extend(patch.get("errors", []))
            answer = patch.get("final_answer")
            if isinstance(answer, BaseModel):
                final_answer = answer.model_dump()
            yield AnalysisEvent(type="progress", node=node_name, data=_serialize_patch(patch))

    yield AnalysisEvent(
        type="complete",
        data={"final_answer": final_answer, "errors": errors},
    )
