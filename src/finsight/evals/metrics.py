"""
Métricas de avaliação de RAG — as 4 do estilo RAGAS, implementadas localmente.

Cada métrica devolve um número em [0, 1] (1 = melhor) e responde a UMA pergunta
de qualidade. Recapitulando o mapa fixado em `dataset.py`:

    | métrica            | camada    | usa do runtime      | usa do gabarito        |
    |--------------------|-----------|---------------------|------------------------|
    | faithfulness       | geração   | answer + contexts   | (nenhum)               |
    | answer_relevancy   | geração   | question + answer   | (nenhum)               |
    | context_precision  | retrieval | retrieved_contexts  | ground_truth_contexts  |
    | context_recall     | retrieval | retrieved_contexts  | ground_truth           |

>>> A grande sacada: 4 métricas, 2 padrões <<<

Padrão A — "decompor um texto em afirmações atômicas e checar cada uma contra um
conjunto de trechos" (LLM-as-judge). Cobre DUAS métricas, que são simétricas:
    - faithfulness:    decompõe a RESPOSTA   -> cada claim sai dos CONTEXTOS RECUPERADOS?
    - context_recall:  decompõe o GABARITO   -> cada claim era recuperável dos CONTEXTOS?
Mesma mecânica (`_classify_claims`), só trocam o que se decompõe e contra o quê.

Padrão B — métricas que NÃO são "claims contra contexto":
    - context_precision: o juiz rotula cada chunk recuperado (relevante?) e a NOTA
      sai de uma fórmula determinística de ranking (Average Precision) — premia o
      relevante no TOPO. O LLM só rotula; a matemática do rank é código puro.
    - answer_relevancy: método clássico da RAGAS — o juiz gera "perguntas-reversas"
      a partir da resposta, EMBEDAMOS essas perguntas e a query original, e tiramos
      o cosseno médio. Resposta no alvo -> perguntas-reversas parecidas com a
      original -> cosseno alto. Reusa `embed_texts` (zero dependência nova).

Decisões de engenharia (consistentes com reranker.py/generator.py):
  - `_get_judge_client()` com lru_cache e temperature=0.0 (julgamento determinístico)
    é o ÚNICO ponto de mock do LLM. Como cada métrica precisa de um schema diferente,
    o client CRU é o mock e cada uma faz `.with_structured_output(Schema)` na hora.
  - Guards determinísticos (resposta vazia, sem contexto) curto-circuitam SEM LLM —
    diagnóstico certo, barato, e CI sem rede.
  - Cada métrica devolve `MetricResult(score, details)`: o runner do Passo 4 lê
    `.score`; quem precisa entender POR QUE a nota foi aquela lê `.details`.

Como o resto do retrieval, este módulo PROPAGA exceções — a tradução para
`state["errors"]` é responsabilidade do nó do grafo (Semana 6).
"""

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from finsight.db.session import settings
from finsight.ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado uniforme de toda métrica
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """
    O retorno de qualquer métrica: a nota + os detalhes que a justificam.

    `score` é sempre [0, 1] (1 = melhor) — é o que o runner agrega na tabela
    comparativa. `details` carrega a evidência (claims julgados, rótulos por
    chunk, perguntas-reversas) para você AUDITAR uma nota ruim sem re-rodar: uma
    média de 0,6 não diz nada; ver "3 de 5 claims sem suporte" diz tudo.
    """

    score: float
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schemas de structured output (um por tipo de julgamento)
# ---------------------------------------------------------------------------


class _ClaimVerdict(BaseModel):
    """Uma afirmação atômica extraída de um texto + se os trechos a sustentam."""

    claim: str = Field(description="Afirmação atômica (um único fato verificável).")
    supported: bool = Field(
        description="True se os TRECHOS sustentam/atribuem esta afirmação; False caso contrário."
    )


class _ClaimAnalysis(BaseModel):
    """Saída do Padrão A: o texto decomposto em claims, cada um julgado."""

    claims: list[_ClaimVerdict] = Field(
        description="Afirmações atômicas do texto, com o veredito de suporte de cada uma."
    )


class _ContextVerdict(BaseModel):
    """Rótulo de relevância de um chunk recuperado, referenciado pelo índice enviado."""

    index: int = Field(description="Índice (0-based) do trecho recuperado na lista enviada.")
    relevant: bool = Field(description="True se o trecho é relevante para responder à pergunta.")


class _ContextAnalysis(BaseModel):
    """Saída de context_precision: um veredito de relevância por chunk recuperado."""

    verdicts: list[_ContextVerdict] = Field(
        description="Veredito de relevância para cada trecho recuperado, por índice."
    )


class _ReverseQuestions(BaseModel):
    """Saída de answer_relevancy: perguntas que a resposta responderia + flag de evasão."""

    questions: list[str] = Field(
        description="Perguntas que esta resposta responde de forma completa."
    )
    noncommittal: bool = Field(
        description=(
            "True se a resposta é evasiva/não-comprometida (ex: 'não sei', 'não há "
            "informação suficiente'); False se afirma algo concreto."
        )
    )


# ---------------------------------------------------------------------------
# Client de juiz (ponto de mock)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_judge_client() -> ChatOpenAI:
    """
    Client de julgamento CRU (sem structured output), criado uma vez. Ponto de mock.

    Diferença DELIBERADA para o reranker: lá o ponto de mock é o runnable JÁ
    embrulhado por `.with_structured_output`, porque o reranker só tem um schema.
    Aqui há QUATRO schemas diferentes (_ClaimAnalysis, _ContextAnalysis, ...), então
    expomos o client cru e cada métrica embrulha com o schema que precisa em `_judge`.
    O teste mocka este client com um fake cujo `.with_structured_output(schema)`
    devolve um runnable determinístico.

    temperature=0.0: avaliar é julgar — queremos a mesma nota para a mesma entrada
    (reprodutibilidade), igual ao reranker e ao generator.
    """
    return ChatOpenAI(
        api_key=SecretStr(settings.openai_api_key),
        model=settings.active_llm_model,
        temperature=0.0,
        max_retries=4,
        timeout=60.0,
    )


async def _judge(
    messages: list[SystemMessage | HumanMessage],
    schema: type[BaseModel],
) -> BaseModel:
    """
    Embrulha o client cru com `schema` e invoca — a ponte entre 1 client e N schemas.

    Centralizar aqui mantém cada métrica enxuta (monta prompt -> chama `_judge`) e
    dá um único lugar onde o `.with_structured_output` acontece.
    """
    client = _get_judge_client()
    structured: Runnable[LanguageModelInput, dict[str, Any] | BaseModel] = (
        client.with_structured_output(schema)
    )
    result = await structured.ainvoke(messages)
    # with_structured_output garante o tipo; o assert é só para o mypy, que vê o
    # retorno genérico do Runnable (mesma defesa do reranker).
    assert isinstance(result, schema)
    return result


def _format_numbered(items: list[str]) -> str:
    """Numera trechos no prompt (0-based) — mesma convenção do reranker/generator."""
    return "\n\n".join(f"[{i}] {item}" for i, item in enumerate(items))


# ===========================================================================
# PADRÃO A — decompor em claims e checar suporte (faithfulness + context_recall)
# ===========================================================================


async def _classify_claims(
    *,
    source_text: str,
    contexts: list[str],
    system_prompt: str,
) -> _ClaimAnalysis:
    """
    Núcleo do Padrão A: decompõe `source_text` em claims e julga cada um contra
    `contexts`, numa única chamada de LLM.

    RAGAS faz isso em duas etapas (decompor; depois NLI por claim). Fazemos numa
    chamada só — mais barato e suficiente para a escala do golden set — pedindo ao
    juiz que devolva os claims JÁ com o veredito de suporte. O `system_prompt`
    muda o que decompor e contra o quê comparar; a mecânica é idêntica.
    """
    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"TEXTO:\n{source_text}\n\nTRECHOS:\n{_format_numbered(contexts)}"),
    ]
    result = await _judge(messages, _ClaimAnalysis)
    assert isinstance(result, _ClaimAnalysis)
    return result


def _claim_fraction(analysis: _ClaimAnalysis) -> float:
    """Fração de claims sustentados. Sem claims -> 0.0 (nada verificável = sem mérito)."""
    if not analysis.claims:
        return 0.0
    supported = sum(1 for c in analysis.claims if c.supported)
    return supported / len(analysis.claims)


_FAITHFULNESS_SYSTEM_PROMPT = (
    "Você avalia FAITHFULNESS (fidelidade) de uma resposta de RAG. Decomponha o "
    "TEXTO (a resposta gerada) em afirmações atômicas — cada uma um único fato "
    "verificável. Para cada afirmação, marque supported=True se ela puder ser "
    "DEDUZIDA diretamente dos TRECHOS fornecidos, e supported=False se ela não "
    "estiver nos trechos (conhecimento externo ou invenção). Seja rigoroso: na "
    "dúvida sobre se um número ou período está nos trechos, marque False."
)


async def faithfulness(answer: str, contexts: list[str]) -> MetricResult:
    """
    Mede se a RESPOSTA está ancorada nos CONTEXTOS (não alucina). [0, 1], 1 = ótimo.

    Score = fração dos claims da resposta que os contextos sustentam. É o espelho
    exato da disciplina de ancoragem do generator: um generator que respeita
    "responda só do contexto" produz faithfulness alto; um que completa com
    conhecimento do modelo produz claims sem suporte -> nota baixa. É a métrica
    anti-alucinação, crítica num agente financeiro.

    Guards (sem LLM): resposta vazia -> 0.0 (nada ancorado); sem contextos -> 0.0
    (não há no que ancorar, logo qualquer afirmação é, por definição, sem suporte).
    """
    if not answer.strip():
        return MetricResult(0.0, {"reason": "resposta vazia"})
    if not contexts:
        return MetricResult(0.0, {"reason": "nenhum contexto para ancorar"})

    analysis = await _classify_claims(
        source_text=answer,
        contexts=contexts,
        system_prompt=_FAITHFULNESS_SYSTEM_PROMPT,
    )
    score = _claim_fraction(analysis)
    return MetricResult(
        score,
        {"claims": [c.model_dump() for c in analysis.claims]},
    )


_CONTEXT_RECALL_SYSTEM_PROMPT = (
    "Você avalia CONTEXT RECALL de um retriever de RAG. Decomponha o TEXTO (a "
    "resposta de referência / gabarito) em afirmações atômicas. Para cada uma, "
    "marque supported=True se a afirmação puder ser ATRIBUÍDA aos TRECHOS "
    "recuperados (a informação está presente neles), e supported=False se os "
    "trechos recuperados NÃO contêm aquilo. Isto mede se o retriever trouxe tudo "
    "o que era necessário para reconstruir o gabarito."
)


async def context_recall(contexts: list[str], ground_truth: str) -> MetricResult:
    """
    Mede se o retriever trouxe TUDO que o gabarito exige. [0, 1], 1 = ótimo.

    Score = fração dos claims do `ground_truth` que são atribuíveis aos `contexts`
    recuperados. Recall baixo = o chunk decisivo ficou de fora (falha de
    recuperação) — é exatamente o que HyDE e o over-fetch do re-rank tentam curar,
    e por isso esta métrica é a que valida o ganho deles no Passo 4.

    Guards (sem LLM): gabarito vazio -> 1.0 (nada a recuperar, recall trivialmente
    perfeito); sem contextos -> 0.0 (recall zero — não trouxe nada).
    """
    if not ground_truth.strip():
        return MetricResult(1.0, {"reason": "gabarito vazio — nada a recuperar"})
    if not contexts:
        return MetricResult(0.0, {"reason": "nenhum contexto recuperado"})

    analysis = await _classify_claims(
        source_text=ground_truth,
        contexts=contexts,
        system_prompt=_CONTEXT_RECALL_SYSTEM_PROMPT,
    )
    score = _claim_fraction(analysis)
    return MetricResult(
        score,
        {"claims": [c.model_dump() for c in analysis.claims]},
    )


# ===========================================================================
# PADRÃO B.1 — context_precision (LLM rotula, Average Precision pontua)
# ===========================================================================


def _average_precision(relevances: list[bool]) -> float:
    """
    Average Precision: premia ter os relevantes NO TOPO da lista recuperada.

    Para cada posição k (1-based) que é relevante, calcula a precisão@k (quantos
    relevantes nas k primeiras posições / k) e tira a média dessas precisões sobre
    o total de relevantes. Um relevante na posição 1 contribui 1/1; o mesmo
    relevante na posição 5, atrás de 4 lixos, contribui só 1/5. É isto que separa
    "trouxe o certo no topo" de "trouxe o certo enterrado" — algo que uma simples
    fração de relevantes NÃO captura.

        relevances = [True, False, True]
        -> k=1: 1/1 (rel) ; k=2: pula (não-rel) ; k=3: 2/3 (rel)
        -> AP = (1/1 + 2/3) / 2 relevantes = 0.833

    Sem nenhum relevante -> 0.0 (precisão indefinida vira a pior nota).
    """
    total_relevant = sum(relevances)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for k, rel in enumerate(relevances, start=1):
        if rel:
            hits += 1
            precision_sum += hits / k
    return precision_sum / total_relevant


_CONTEXT_PRECISION_SYSTEM_PROMPT = (
    "Você avalia a RELEVÂNCIA de trechos recuperados por um RAG. Receberá a "
    "PERGUNTA, os TRECHOS DE REFERÊNCIA (o que um retriever ideal deveria trazer) "
    "e os TRECHOS RECUPERADOS (o que o sistema trouxe), numerados por índice. Para "
    "CADA trecho recuperado, marque relevant=True se ele ajuda a responder à "
    "pergunta ou contém informação presente na referência, e relevant=False se for "
    "tangencial, genérico ou off-topic. Devolva um veredito para cada índice."
)


async def context_precision(
    question: str,
    retrieved_contexts: list[str],
    ground_truth_contexts: list[str],
) -> MetricResult:
    """
    Mede se os chunks RELEVANTES vieram no TOPO da recuperação. [0, 1], 1 = ótimo.

    Duas etapas: (1) o juiz rotula cada chunk recuperado como relevante ou não,
    usando `ground_truth_contexts` como referência do que importa; (2) calculamos a
    Average Precision sobre esses rótulos NA ORDEM recuperada — código puro,
    determinístico e testável sem mock. Ordem importa porque só os primeiros chunks
    entram no contexto do LLM; relevante na posição 8 que foi cortado no top 5 é
    inútil mesmo tendo sido "recuperado".

    Guard (sem LLM): sem contextos recuperados -> 0.0 (não trouxe nada).
    """
    if not retrieved_contexts:
        return MetricResult(0.0, {"reason": "nenhum contexto recuperado"})

    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=_CONTEXT_PRECISION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"PERGUNTA:\n{question}\n\n"
                f"TRECHOS DE REFERÊNCIA:\n{_format_numbered(ground_truth_contexts)}\n\n"
                f"TRECHOS RECUPERADOS:\n{_format_numbered(retrieved_contexts)}"
            )
        ),
    ]
    result = await _judge(messages, _ContextAnalysis)
    assert isinstance(result, _ContextAnalysis)

    # Remonta os rótulos NA ORDEM recuperada. O juiz pode omitir índices, repetir
    # ou alucinar fora de range; mapeamos por índice e tratamos ausente como
    # não-relevante (conservador: o que não foi afirmado relevante não pontua).
    relevant_by_index = {
        v.index: v.relevant for v in result.verdicts if 0 <= v.index < len(retrieved_contexts)
    }
    relevances = [relevant_by_index.get(i, False) for i in range(len(retrieved_contexts))]

    score = _average_precision(relevances)
    return MetricResult(
        score,
        {"relevances": relevances},
    )


# ===========================================================================
# PADRÃO B.2 — answer_relevancy (perguntas-reversas + cosseno de embeddings)
# ===========================================================================

_ANSWER_RELEVANCY_SYSTEM_PROMPT = (
    "Dada uma RESPOSTA, gere as perguntas que ela responde de forma completa e "
    "direta (gere de 1 a {n} perguntas, variando a formulação). Marque também "
    "noncommittal=True se a resposta for evasiva ou não-comprometida — por exemplo "
    "'não sei', 'não há informação suficiente', 'não posso responder' — e "
    "noncommittal=False se ela afirmar algo concreto."
)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosseno entre dois vetores. Vetor nulo -> 0.0 (sem direção, sem similaridade)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def answer_relevancy(
    question: str,
    answer: str,
    *,
    n_questions: int = 3,
) -> MetricResult:
    """
    Mede se a resposta DE FATO responde à pergunta, sem enrolar. [0, 1], 1 = ótimo.

    Método clássico da RAGAS: se a resposta é boa, dá para "adivinhar" a pergunta
    original a partir dela. Então pedimos ao LLM `n_questions` perguntas-reversas
    geradas SÓ a partir da resposta, embedamos essas perguntas e a query original
    (reusando `embed_texts`, mesmo modelo da busca) e tiramos o cosseno médio.
    Resposta no alvo -> perguntas-reversas semanticamente próximas da original ->
    cosseno alto. Resposta que fala de outra coisa -> perguntas distantes -> baixo.

    O flag `noncommittal` zera a nota: uma recusa ("não há informação suficiente")
    pode até gerar perguntas-reversas plausíveis, mas NÃO responde à pergunta —
    sem este detector, uma evasão bem-redigida ganharia nota imerecida. (Note a
    complementaridade: a mesma recusa que dá faithfulness alto — não alucina — dá
    answer_relevancy zero. Juntas, as duas métricas localizam que o problema foi
    de RETRIEVAL, não de geração.)

    Guard (sem LLM): resposta vazia -> 0.0.
    """
    if not answer.strip():
        return MetricResult(0.0, {"reason": "resposta vazia"})

    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=_ANSWER_RELEVANCY_SYSTEM_PROMPT.format(n=n_questions)),
        HumanMessage(content=f"RESPOSTA:\n{answer}"),
    ]
    result = await _judge(messages, _ReverseQuestions)
    assert isinstance(result, _ReverseQuestions)

    # Evasão: a resposta não se compromete -> não responde à pergunta -> nota 0,
    # independentemente de quão plausíveis foram as perguntas-reversas.
    if result.noncommittal:
        return MetricResult(0.0, {"reason": "resposta evasiva (noncommittal)"})
    if not result.questions:
        return MetricResult(0.0, {"reason": "nenhuma pergunta-reversa gerada"})

    # Uma única chamada de embeddings com a query na posição 0 e as reversas em
    # seguida — preserva a ordem (garantia do embed_texts) e economiza round-trips.
    vectors = await embed_texts([question, *result.questions])
    query_vec, reverse_vecs = vectors[0], vectors[1:]

    similarities = [_cosine(query_vec, rv) for rv in reverse_vecs]
    score = sum(similarities) / len(similarities)
    # Cosseno pode dar levemente negativo; clamp em [0, 1] mantém a escala da métrica.
    score = max(0.0, min(1.0, score))
    return MetricResult(
        score,
        {"reverse_questions": result.questions, "similarities": similarities},
    )
