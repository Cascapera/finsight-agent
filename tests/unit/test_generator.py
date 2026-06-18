"""
Testes do generator (Passo 2 da Semana 4) — sem banco, sem rede.

Mock: substituímos `generator._get_answer_client` por um fake cujo `.ainvoke`
devolve um objeto com `.content`. Dois fakes:
  * _EchoClient: captura as mensagens recebidas (para inspecionar o prompt) e
    devolve um conteúdo controlado.
  * _ExplodingClient: falha se chamado — prova que o guard de contexto vazio NÃO
    invoca o LLM.
"""

from typing import Any

import pytest
from langchain_core.messages import SystemMessage

from finsight.evals import generator
from finsight.evals.generator import _NO_CONTEXT_ANSWER, generate_answer


class _EchoClient:
    """Guarda as mensagens recebidas e devolve um `.content` fixo."""

    def __init__(self, content: Any) -> None:
        self._content = content
        self.last_messages: Any = None

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.last_messages = messages
        # Espelha a interface do ChatOpenAI: response.content.
        return type("FakeResponse", (), {"content": self._content})()


class _ExplodingClient:
    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        raise AssertionError("o LLM não deveria ter sido chamado para contexto vazio")


@pytest.mark.asyncio
async def test_generate_answer_grounds_on_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resposta vem do LLM e o prompt enviado contém os trechos (ancoragem)."""
    fake = _EchoClient("A receita foi de R$ 48,2 bilhões.")
    monkeypatch.setattr(generator, "_get_answer_client", lambda: fake)

    contexts = ["A receita líquida atingiu R$ 48,2 bilhões em 2024."]
    answer = await generate_answer("Qual a receita?", contexts)

    assert answer == "A receita foi de R$ 48,2 bilhões."
    # O system prompt impõe a ancoragem...
    assert isinstance(fake.last_messages[0], SystemMessage)
    assert "EXCLUSIVAMENTE" in fake.last_messages[0].content
    # ...e o trecho de contexto chega de fato ao LLM (mensagem humana).
    assert "R$ 48,2 bilhões" in fake.last_messages[-1].content
    assert "Qual a receita?" in fake.last_messages[-1].content


@pytest.mark.asyncio
async def test_empty_contexts_short_circuits_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem contexto: recusa determinística, e o LLM NÃO é chamado."""
    monkeypatch.setattr(generator, "_get_answer_client", _ExplodingClient)

    answer = await generate_answer("Qual a receita?", [])

    assert answer == _NO_CONTEXT_ANSWER


@pytest.mark.asyncio
async def test_normalizes_non_string_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """content multimodal (lista) é normalizado para str — métricas esperam texto."""
    fake = _EchoClient(["pedaço A", "pedaço B"])
    monkeypatch.setattr(generator, "_get_answer_client", lambda: fake)

    answer = await generate_answer("q", ["algum contexto"])

    assert isinstance(answer, str)
    assert "pedaço A" in answer
