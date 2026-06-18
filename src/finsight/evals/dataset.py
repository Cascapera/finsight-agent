"""
Golden dataset — o contrato de dados da avaliação de RAG.

Avaliar RAG é um problema supervisionado: para dizer se o sistema acertou, você
precisa de um GABARITO. Este módulo define o formato desse gabarito e carrega o
conjunto de avaliação ("golden set").

>>> O conceito central: o que cada métrica consome <<<

Um pipeline de RAG produz, para cada pergunta, DOIS artefatos em runtime:
    - retrieved_contexts -> os chunks que o retriever trouxe   (varia por config)
    - generated_answer   -> a resposta que o LLM gerou          (varia por config)

E o golden set fornece os DOIS lados do gabarito, fixos:
    - ground_truth          -> a resposta de referência (correta)
    - ground_truth_contexts -> os trechos que DEVERIAM ter sido recuperados

As 4 métricas da Semana 4 cruzam runtime x gabarito assim:

    | métrica            | camada     | usa do runtime        | usa do gabarito          |
    |--------------------|------------|-----------------------|--------------------------|
    | context_recall     | retrieval  | retrieved_contexts    | ground_truth             |
    | context_precision  | retrieval  | retrieved_contexts    | ground_truth_contexts    |
    | faithfulness       | geração    | answer + contexts     | (nenhum — auto-contido)  |
    | answer_relevancy   | geração    | question + answer     | (nenhum)                 |

Por isso o EvalSample carrega `ground_truth` E `ground_truth_contexts`: são
gabaritos de camadas diferentes (resposta certa vs. recuperação certa), e
métricas diferentes consomem cada um. Faithfulness e answer_relevancy não usam
gabarito nenhum — são "auto-contidas" (medem coerência interna), por isso dá pra
rodá-las em produção sem golden set; as de retrieval exigem o gabarito e ficam
restritas ao conjunto de avaliação.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvalSample(BaseModel):
    """
    Um caso de avaliação: uma pergunta e seu gabarito completo.

    É deliberadamente um BaseModel estrito (mesma convenção dos outputs em
    state.py): o golden set é dado de entrada e queremos ValidationError
    explícito se alguém escrever um sample malformado, não coerção silenciosa.
    """

    model_config = ConfigDict(strict=True)

    # Identificador estável do caso. Aparece no relatório comparativo (Passo 4)
    # para você rastrear QUAL pergunta regrediu entre baseline/HyDE/rerank — uma
    # média esconde o caso individual que quebrou.
    id: str = Field(description="Identificador estável do caso (ex: 'q-receita-2024').")

    # A pergunta em linguagem natural — o input de todo o pipeline e de toda
    # métrica.
    question: str = Field(description="Pergunta do usuário em linguagem natural.")

    # A resposta de referência (o que um humano expert responderia).
    # Consumida por context_recall: a métrica DECOMPÕE esta resposta em claims e
    # checa se cada claim era recuperável do contexto. Logo, escreva o
    # ground_truth de forma factual e enxuta — claims claros geram recall
    # confiável; prosa vaga gera ruído.
    ground_truth: str = Field(description="Resposta de referência (gabarito factual).")

    # Os trechos que um retriever IDEAL deveria trazer para responder.
    # Consumido por context_precision: serve de referência para julgar se cada
    # chunk recuperado é "relevante". Não precisa ser o texto literal do chunk no
    # banco — descreve o CONTEÚDO que importa; o juiz compara semanticamente.
    ground_truth_contexts: list[str] = Field(
        default_factory=list,
        description="Trechos que o retriever ideal deveria recuperar.",
    )

    # Filtro opcional de ativo, repassado ao retriever (retrieve(..., ticker=)).
    # Mantém a busca restrita ao documento certo, espelhando o uso real do RAG
    # Agent. None = busca em todo o corpus.
    ticker: str | None = Field(
        default=None, description="Ticker para filtrar a busca; None = corpus inteiro."
    )


class EvalDataset(BaseModel):
    """
    Coleção de EvalSample, com I/O em JSON e filtro por ticker.

    Por que um wrapper e não uma `list[EvalSample]` solta? Para (a) ter um ponto
    único de load/save (o golden set cresce e vai morar em disco/versionado) e
    (b) expor helpers (`__iter__`, `__len__`, `filter_by_ticker`) sem espalhar
    `dataset.samples` por todo lado. O runner do Passo 4 itera sobre isto.
    """

    model_config = ConfigDict(strict=True)

    samples: list[EvalSample] = Field(default_factory=list)

    # Nota: NÃO sobrescrevemos __iter__ — BaseModel.__iter__ tem semântica
    # própria (itera pares campo/valor) e brigar com ela exige type: ignore.
    # Iteramos `dataset.samples` explicitamente, que é mais claro de qualquer modo.
    def __len__(self) -> int:
        return len(self.samples)

    def filter_by_ticker(self, ticker: str) -> "EvalDataset":
        """Subconjunto dos casos de um ativo — útil para avaliar um doc por vez."""
        return EvalDataset(samples=[s for s in self.samples if s.ticker == ticker])

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalDataset":
        """Carrega de um arquivo JSON (lista de objetos sample ou {"samples": [...]})."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Aceita as duas formas comuns: o arquivo é a lista direta, ou um objeto
        # com a chave "samples". Tolerância barata que evita atrito ao versionar.
        if isinstance(raw, list):
            raw = {"samples": raw}
        return cls.model_validate(raw)

    def to_json(self, path: str | Path) -> None:
        """Salva em JSON legível (indentado, UTF-8 preservado para acentos)."""
        Path(path).write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# SEED — golden set de exemplo, auto-contido e coerente.
#
# ATENÇÃO: estes casos descrevem um relatório FICTÍCIO ("Petro Norte") só para
# (a) o pipeline rodar de ponta a ponta sem depender de um PDF específico e
# (b) servir de fixture determinística nos testes. Em uso real, SUBSTITUA por
# perguntas ancoradas nos seus PDFs ingeridos — o golden set só vale o quanto
# espelha o corpus de verdade. Os fatos abaixo são internamente consistentes:
# os ground_truth são deriváveis dos ground_truth_contexts.
# ---------------------------------------------------------------------------
SEED_SAMPLES: list[EvalSample] = [
    EvalSample(
        id="q-receita-2024",
        question="Qual foi a receita líquida da Petro Norte em 2024 e como variou?",
        ground_truth=(
            "A receita líquida da Petro Norte em 2024 foi de R$ 48,2 bilhões, "
            "um crescimento de 12% em relação aos R$ 43,0 bilhões de 2023."
        ),
        ground_truth_contexts=[
            "A receita líquida consolidada atingiu R$ 48,2 bilhões em 2024, "
            "ante R$ 43,0 bilhões em 2023, alta de 12% no comparativo anual.",
        ],
        ticker="PNOR3",
    ),
    EvalSample(
        id="q-margem-ebitda",
        question="Qual foi a margem EBITDA da Petro Norte em 2024?",
        ground_truth="A margem EBITDA da Petro Norte em 2024 foi de 31%.",
        ground_truth_contexts=[
            "O EBITDA ajustado somou R$ 14,9 bilhões, equivalente a uma margem "
            "EBITDA de 31% sobre a receita líquida do exercício de 2024.",
        ],
        ticker="PNOR3",
    ),
    EvalSample(
        id="q-divida-liquida",
        question="Qual era a dívida líquida e a alavancagem da Petro Norte no fim de 2024?",
        ground_truth=(
            "Ao fim de 2024 a dívida líquida da Petro Norte era de R$ 22,4 bilhões, "
            "com alavancagem de 1,5x dívida líquida/EBITDA."
        ),
        ground_truth_contexts=[
            "A dívida líquida encerrou 2024 em R$ 22,4 bilhões. A relação "
            "dívida líquida/EBITDA ficou em 1,5x, ante 1,8x no fim de 2023.",
        ],
        ticker="PNOR3",
    ),
]

SEED_DATASET = EvalDataset(samples=SEED_SAMPLES)
