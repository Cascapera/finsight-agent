"""
Testes da instrumentação Prometheus (Semana 7, Passo 1) — sem rede.

As métricas vivem no registry GLOBAL do prometheus_client (compartilhado entre
testes), então medimos DELTAS (valor depois - valor antes), não valores absolutos.
Lemos os contadores/histograma via a API pública `registry.get_sample_value`.
"""

from typing import Any

import pytest
from prometheus_client import REGISTRY

from finsight.observability.metrics import instrument_node


def _runs(node: str) -> float:
    return REGISTRY.get_sample_value("finsight_node_runs_total", {"node": node}) or 0.0


def _errors(node: str) -> float:
    return REGISTRY.get_sample_value("finsight_node_errors_total", {"node": node}) or 0.0


def _duration_count(node: str) -> float:
    # O Histogram expõe um _count com o número de observações.
    return REGISTRY.get_sample_value("finsight_node_duration_seconds_count", {"node": node}) or 0.0


@pytest.mark.asyncio
async def test_instrument_counts_success() -> None:
    """Nó de sucesso: runs +1, duração observada +1, erros inalterado."""
    node = "test_success"

    async def ok(_state: Any) -> dict[str, Any]:
        return {"financial": "stub"}

    before_runs, before_errors, before_dur = _runs(node), _errors(node), _duration_count(node)
    await instrument_node(node, ok)({"ticker": "X"})  # type: ignore[arg-type]

    assert _runs(node) == before_runs + 1
    assert _duration_count(node) == before_dur + 1
    assert _errors(node) == before_errors  # sem erro


@pytest.mark.asyncio
async def test_instrument_counts_error_from_patch() -> None:
    """
    Nó que degrada (patch com `errors`): erros +1. Lemos o erro do CONTRATO do patch,
    não de exceção — é como nossos nós sinalizam falha.
    """
    node = "test_degraded"

    async def degraded(_state: Any) -> dict[str, Any]:
        return {"financial": "stub", "errors": ["boom"]}

    before_runs, before_errors = _runs(node), _errors(node)
    await instrument_node(node, degraded)({"ticker": "X"})  # type: ignore[arg-type]

    assert _runs(node) == before_runs + 1
    assert _errors(node) == before_errors + 1


@pytest.mark.asyncio
async def test_instrument_reraises_and_counts_exception() -> None:
    """Defensivo: nó que LEVANTA -> conta erro e re-levanta (não mascara o bug)."""
    node = "test_raises"

    async def boom(_state: Any) -> dict[str, Any]:
        raise RuntimeError("unexpected")

    before_errors, before_dur = _errors(node), _duration_count(node)
    with pytest.raises(RuntimeError, match="unexpected"):
        await instrument_node(node, boom)({"ticker": "X"})  # type: ignore[arg-type]

    assert _errors(node) == before_errors + 1
    # latência medida mesmo no caminho de exceção (finally).
    assert _duration_count(node) == before_dur + 1
