"""
Testes das 4 métricas (Passo 3 da Semana 4) — sem banco, sem rede.

Estratégia de mock, espelhando o resto do projeto:
  * O LLM: substituímos `metrics._get_judge_client` por um fake cujo
    `.with_structured_output(schema)` devolve um runnable determinístico. O fake
    inspeciona o prompt e fabrica uma instância do `schema` pedido — assim um
    único fake serve às quatro métricas (cada uma usa um schema diferente).
  * Os embeddings (answer_relevancy): substituímos `metrics.embed_texts` por um
    fake que mapeia palavras-chave -> dimensões ORTOGONAIS (mesma técnica do
    test_retriever.py), tornando o cosseno previsível e o teste determinístico.

Também testamos `_average_precision` e `_cosine` diretamente: são matemática pura,
o coração de duas métricas, e merecem teste sem nenhum mock.
"""

from typing import Any

import pytest

from finsight.evals import metrics
from finsight.evals.metrics import (
    MetricResult,
    _average_precision,
    _ClaimAnalysis,
    _ClaimVerdict,
    _ContextAnalysis,
    _ContextVerdict,
    _cosine,
    _ReverseQuestions,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# ---------------------------------------------------------------------------
# Fakes do client de juiz
# ---------------------------------------------------------------------------


class _StructuredRunnable:
    """Runnable fake: guarda o schema pedido e delega a produção ao `responder`."""

    def __init__(self, schema: type, responder: Any) -> None:
        self._schema = schema
        self._responder = responder
        self.last_messages: Any = None

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.last_messages = messages
        return self._responder(self._schema, messages)


class _FakeJudgeClient:
    """
    Fake do client cru: `.with_structured_output(schema)` -> _StructuredRunnable.

    Recebe um `responder(schema, messages) -> BaseModel` que cada teste customiza
    para devolver o veredito determinístico que quer exercitar.
    """

    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.last_runnable: _StructuredRunnable | None = None

    def with_structured_output(self, schema: type) -> _StructuredRunnable:
        self.last_runnable = _StructuredRunnable(schema, self._responder)
        return self.last_runnable


class _ExplodingJudgeClient:
    """Falha se usado — prova que um guard curto-circuitou ANTES de chamar o LLM."""

    def with_structured_output(self, schema: type) -> Any:
        raise AssertionError("o LLM de juiz não deveria ter sido chamado")


def _patch_judge(monkeypatch: pytest.MonkeyPatch, responder: Any) -> _FakeJudgeClient:
    fake = _FakeJudgeClient(responder)
    monkeypatch.setattr(metrics, "_get_judge_client", lambda: fake)
    return fake


# ===========================================================================
# Matemática pura — sem mocks
# ===========================================================================


@pytest.mark.parametrize(
    "relevances, expected",
    [
        ([True, True, True], 1.0),  # todos relevantes no topo -> perfeito
        ([False, False, False], 0.0),  # nenhum relevante -> zero
        ([], 0.0),  # vazio -> zero
        ([True, False, True], (1 / 1 + 2 / 3) / 2),  # 0.8333...
        ([False, True], (1 / 2) / 1),  # relevante na pos 2 -> penalizado
    ],
)
def test_average_precision(relevances: list[bool], expected: float) -> None:
    assert _average_precision(relevances) == pytest.approx(expected)


def test_average_precision_rewards_top_rank() -> None:
    """O MESMO relevante pontua mais no topo do que enterrado — o ponto da métrica."""
    top = _average_precision([True, False, False, False])
    buried = _average_precision([False, False, False, True])
    assert top > buried


def test_cosine() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)  # idênticos
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)  # ortogonais
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)  # vetor nulo


# ===========================================================================
# faithfulness
# ===========================================================================


@pytest.mark.asyncio
async def test_faithfulness_fraction_of_supported_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score = claims sustentados / total. 2 de 3 -> 0.666..."""

    def responder(schema: type, messages: Any) -> Any:
        assert schema is _ClaimAnalysis
        return _ClaimAnalysis(
            claims=[
                _ClaimVerdict(claim="receita R$ 48,2 bi", supported=True),
                _ClaimVerdict(claim="margem 31%", supported=True),
                _ClaimVerdict(claim="CEO trocou", supported=False),
            ]
        )

    fake = _patch_judge(monkeypatch, responder)
    result = await faithfulness("alguma resposta", ["algum contexto"])

    assert isinstance(result, MetricResult)
    assert result.score == pytest.approx(2 / 3)
    assert len(result.details["claims"]) == 3
    # O prompt de fato carrega a resposta e o contexto ao juiz.
    assert "alguma resposta" in fake.last_runnable.last_messages[-1].content
    assert "algum contexto" in fake.last_runnable.last_messages[-1].content


@pytest.mark.asyncio
async def test_faithfulness_empty_answer_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resposta vazia -> 0.0 sem chamar o LLM."""
    monkeypatch.setattr(metrics, "_get_judge_client", _ExplodingJudgeClient)
    result = await faithfulness("   ", ["contexto"])
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_faithfulness_no_context_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem contexto -> 0.0 sem chamar o LLM (nada para ancorar)."""
    monkeypatch.setattr(metrics, "_get_judge_client", _ExplodingJudgeClient)
    result = await faithfulness("uma resposta", [])
    assert result.score == 0.0


# ===========================================================================
# context_recall
# ===========================================================================


@pytest.mark.asyncio
async def test_context_recall_fraction_attributable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score = claims do gabarito atribuíveis ao contexto / total. 1 de 2 -> 0.5."""

    def responder(schema: type, messages: Any) -> Any:
        return _ClaimAnalysis(
            claims=[
                _ClaimVerdict(claim="receita R$ 48,2 bi", supported=True),
                _ClaimVerdict(claim="alta de 12%", supported=False),
            ]
        )

    _patch_judge(monkeypatch, responder)
    result = await context_recall(["contexto recuperado"], "gabarito")
    assert result.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_context_recall_empty_ground_truth_is_perfect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gabarito vazio -> 1.0 sem LLM (nada a recuperar)."""
    monkeypatch.setattr(metrics, "_get_judge_client", _ExplodingJudgeClient)
    result = await context_recall(["contexto"], "  ")
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_context_recall_no_context_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem contexto recuperado -> recall 0.0 sem LLM."""
    monkeypatch.setattr(metrics, "_get_judge_client", _ExplodingJudgeClient)
    result = await context_recall([], "gabarito")
    assert result.score == 0.0


# ===========================================================================
# context_precision
# ===========================================================================


@pytest.mark.asyncio
async def test_context_precision_weights_by_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relevante na posição 0, irrelevante na 1 -> AP = 1.0 (relevante no topo)."""

    def responder(schema: type, messages: Any) -> Any:
        assert schema is _ContextAnalysis
        return _ContextAnalysis(
            verdicts=[
                _ContextVerdict(index=0, relevant=True),
                _ContextVerdict(index=1, relevant=False),
            ]
        )

    _patch_judge(monkeypatch, responder)
    result = await context_precision("pergunta", ["chunk bom", "chunk ruim"], ["referência"])
    assert result.score == pytest.approx(1.0)
    assert result.details["relevances"] == [True, False]


@pytest.mark.asyncio
async def test_context_precision_missing_verdict_is_irrelevant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Índice omitido ou fora de range pelo juiz é tratado como não-relevante."""

    def responder(schema: type, messages: Any) -> Any:
        # Só pontua o índice 1 (relevante) e aluciona um índice 9 fora de range.
        return _ContextAnalysis(
            verdicts=[
                _ContextVerdict(index=1, relevant=True),
                _ContextVerdict(index=9, relevant=True),
            ]
        )

    _patch_judge(monkeypatch, responder)
    result = await context_precision("q", ["chunk0", "chunk1"], ["ref"])
    # Índice 0 ausente -> False; índice 9 ignorado. relevances = [False, True].
    # AP = (precisão@2 = 1/2) / 1 relevante = 0.5.
    assert result.details["relevances"] == [False, True]
    assert result.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_context_precision_no_context_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_get_judge_client", _ExplodingJudgeClient)
    result = await context_precision("q", [], ["ref"])
    assert result.score == 0.0


# ===========================================================================
# answer_relevancy
# ===========================================================================


def _fake_embed_factory() -> Any:
    """
    Embeddings fake: cada palavra-chave vira uma dimensão ortogonal. Frases que
    compartilham palavras ficam próximas no cosseno; sem palavras em comum, ortogonais.
    """
    vocab = {"receita": 0, "margem": 1, "divida": 2, "lucro": 3}

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * len(vocab)
            for word, dim in vocab.items():
                if word in text.lower():
                    vec[dim] = 1.0
            vectors.append(vec)
        return vectors

    return fake_embed


@pytest.mark.asyncio
async def test_answer_relevancy_high_when_reverse_matches_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pergunta-reversa que cobre a mesma palavra-chave da query -> cosseno 1.0."""

    def responder(schema: type, messages: Any) -> Any:
        assert schema is _ReverseQuestions
        return _ReverseQuestions(
            questions=["Qual foi a receita?", "Quanto de receita?"],
            noncommittal=False,
        )

    _patch_judge(monkeypatch, responder)
    monkeypatch.setattr(metrics, "embed_texts", _fake_embed_factory())

    result = await answer_relevancy("Qual a receita da empresa?", "A receita foi alta.")
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_answer_relevancy_low_when_reverse_is_offtopic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Perguntas-reversas sobre outro tema (margem) -> ortogonais à query (receita) -> 0."""

    def responder(schema: type, messages: Any) -> Any:
        return _ReverseQuestions(questions=["Qual a margem?"], noncommittal=False)

    _patch_judge(monkeypatch, responder)
    monkeypatch.setattr(metrics, "embed_texts", _fake_embed_factory())

    result = await answer_relevancy("Qual a receita?", "A margem foi de 31%.")
    assert result.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_answer_relevancy_noncommittal_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resposta evasiva -> 0.0 mesmo gerando perguntas; embeddings nem são chamados."""

    def responder(schema: type, messages: Any) -> Any:
        return _ReverseQuestions(questions=["Qual a receita?"], noncommittal=True)

    _patch_judge(monkeypatch, responder)

    def exploding_embed(texts: list[str]) -> Any:
        raise AssertionError("embed_texts não deveria ser chamado para resposta evasiva")

    monkeypatch.setattr(metrics, "embed_texts", exploding_embed)

    result = await answer_relevancy("Qual a receita?", "Não há informação suficiente.")
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_answer_relevancy_empty_answer_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_get_judge_client", _ExplodingJudgeClient)
    result = await answer_relevancy("Qual a receita?", "   ")
    assert result.score == 0.0
