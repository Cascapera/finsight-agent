"""
Testes do tracing LangSmith (Semana 7, Passo 2) — sem rede.

  * configure_tracing: desligado (default) não mexe no ambiente; ligado exporta as
    env vars que o LangChain lê. Usamos monkeypatch.delenv para que o teardown
    restaure o ambiente mesmo tendo sido escrito direto em os.environ.
  * _run_config: a correlação — run_name/metadata.execution_id/tags montados certos.
"""

import os

import pytest

from finsight.graph.orchestrator import _run_config, build_initial_state
from finsight.observability import tracing


def test_configure_tracing_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing desligado: não exporta nada e retorna False."""
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.setattr(tracing.settings, "langchain_tracing_v2", False)

    assert tracing.configure_tracing() is False
    assert "LANGCHAIN_TRACING_V2" not in os.environ


def test_configure_tracing_enabled_exports_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing ligado: exporta as 3 env vars. delenv garante limpeza no teardown."""
    for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tracing.settings, "langchain_tracing_v2", True)
    monkeypatch.setattr(tracing.settings, "langchain_api_key", "ls-secret")
    monkeypatch.setattr(tracing.settings, "langchain_project", "finsight-test")

    assert tracing.configure_tracing() is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "finsight-test"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-secret"


def test_run_config_correlates_execution_id() -> None:
    """O config carrega execution_id no metadata + run_name e tags úteis."""
    state = build_initial_state("vale a pena?", "PETR4", "fii")
    config = _run_config(state)

    assert config["run_name"] == "finsight-analysis:PETR4"
    assert config["metadata"]["execution_id"] == state["execution_id"]
    assert config["metadata"]["ticker"] == "PETR4"
    assert "finsight" in config["tags"]
    assert "fii" in config["tags"]
