"""
Métricas Prometheus + instrumentação dos nós do grafo.

>>> O que medimos (todas com label `node`) <<<
    finsight_node_duration_seconds  (Histogram) — latência por nó. Histogram para
        termos percentis (p50/p95/p99) no Grafana, não só o último valor.
    finsight_node_runs_total        (Counter)   — execuções por nó.
    finsight_node_errors_total      (Counter)   — execuções que terminaram em erro.

`rate(node_errors_total) / rate(node_runs_total)` dá a taxa de erro por agente — o
SLI que vai pro painel.

>>> Decisão de design: instrumentar no ORQUESTRADOR, não dentro dos agentes <<<
`instrument_node(name, fn)` embrulha um nó; o orquestrador registra os nós já
embrulhados. Assim os agentes ficam PUROS (sem dependência de Prometheus) e a
observabilidade é uma cross-cutting concern num só lugar.

>>> Ponto não-óbvio: erro lido do PATCH, não de exceção <<<
Nossos nós NUNCA levantam — degradam e devolvem `{"errors": [...]}` (regra da
Semana 5). Então o wrapper detecta erro inspecionando o patch retornado (tem a
chave `errors`?), não capturando exception. A instrumentação lê o CONTRATO do nó.
Mesmo assim, por robustez, se um nó algum dia levantar, contamos o erro, medimos a
latência no finally e RE-LEVANTAMOS (não engolimos bug de instrumentação).
"""

import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from prometheus_client import Counter, Histogram

from finsight.graph.state import AgentState

logger = logging.getLogger(__name__)

# Tipo de um nó do grafo: async (state) -> patch parcial. Usamos Coroutine (não o
# Awaitable mais amplo) para casar com os overloads de `StateGraph.add_node`, que
# esperam exatamente a assinatura de uma `async def`.
NodeFn = Callable[[AgentState], Coroutine[Any, Any, dict[str, Any]]]

# Buckets do histograma em SEGUNDOS, calibrados para chamadas de LLM/rede (não para
# requests web sub-segundo). Vão de 100ms a 30s — cobre desde um cache hit até uma
# geração lenta do GPT-4o. O bucket +Inf é adicionado automaticamente.
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)

NODE_DURATION = Histogram(
    "finsight_node_duration_seconds",
    "Latência de execução de cada nó do grafo, em segundos.",
    labelnames=("node",),
    buckets=_LATENCY_BUCKETS,
)

NODE_RUNS = Counter(
    "finsight_node_runs_total",
    "Total de execuções de cada nó do grafo.",
    labelnames=("node",),
)

NODE_ERRORS = Counter(
    "finsight_node_errors_total",
    "Total de execuções de cada nó que terminaram em erro.",
    labelnames=("node",),
)


def instrument_node(name: str, fn: NodeFn) -> NodeFn:
    """
    Embrulha um nó do grafo com métricas Prometheus.

    Mede a latência (sempre, via finally), conta a execução, e conta erro quando o
    patch traz a chave `errors` (caminho normal de degradação) OU quando o nó levanta
    (caminho defensivo — contamos e re-levantamos, sem mascarar).
    """

    async def _wrapped(state: AgentState) -> dict[str, Any]:
        NODE_RUNS.labels(node=name).inc()
        start = time.perf_counter()
        try:
            patch = await fn(state)
        except Exception:
            # Defensivo: nossos nós não deveriam chegar aqui. Contamos o erro e
            # re-levantamos — instrumentação nunca esconde um bug do nó.
            NODE_ERRORS.labels(node=name).inc()
            raise
        else:
            # Caminho normal: erro é um item no contrato do patch, não uma exceção.
            if patch.get("errors"):
                NODE_ERRORS.labels(node=name).inc()
            return patch
        finally:
            NODE_DURATION.labels(node=name).observe(time.perf_counter() - start)

    return _wrapped
