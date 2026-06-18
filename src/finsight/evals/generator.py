"""
Generator — o "G" do RAG: transforma (pergunta + chunks) em uma resposta.

Até aqui o projeto só tinha o "R" (retrieval). Mas RAG = Retrieval-AUGMENTED
Generation: depois de recuperar os trechos, um LLM redige a resposta ANCORADA
neles. Este módulo é essa peça de geração.

Por que ela mora em `evals/` (e não num agente ainda)? Porque a Semana 4 precisa
de uma resposta para AVALIAR — faithfulness e answer_relevancy operam sobre o
texto gerado. Este generator é um stand-in enxuto do futuro RAG Agent (Semana 6):
mesma ideia (contexto -> resposta), sem o overhead do nó de grafo. Quando o RAG
Agent existir, ele reusa este mesmo prompt de ancoragem.

>>> O elo com a métrica faithfulness <<<
A regra mais importante do prompt abaixo é "responda APENAS a partir do contexto;
se a informação não estiver lá, diga que não sabe". Isso não é cosmético: é
LITERALMENTE o que faithfulness mede depois (todo claim da resposta tem que sair
do contexto). Um generator que respeita a ancoragem produz faithfulness alto; um
que "completa" com conhecimento prévio do modelo produz alucinação — perigoso num
agente financeiro. Ou seja: a disciplina de geração e a métrica são dois lados da
mesma moeda.

temperature=0.0: queremos a resposta mais provável e determinística dado o
contexto — o oposto do HyDE (0.7, que busca diversidade). Aqui diversidade seria
ruído: a mesma pergunta + mesmos chunks devem dar a mesma resposta (reprodutível
para avaliar).
"""

import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from finsight.db.session import settings

logger = logging.getLogger(__name__)

# Mensagem devolvida quando NÃO há contexto recuperado. É uma recusa explícita,
# determinística e sem chamada de LLM. Importa para a avaliação: se o retriever
# falhou (recall zero), a resposta correta do sistema é admitir que não sabe — e
# uma recusa não introduz claims alucinados (faithfulness fica trivialmente alto,
# answer_relevancy baixo, que é o diagnóstico certo: "falha de retrieval").
_NO_CONTEXT_ANSWER = (
    "Não há contexto recuperado suficiente para responder a esta pergunta com base "
    "nos documentos disponíveis."
)

# A âncora do generator. Espelha a regra de ouro do RAG: o LLM é um REDATOR sobre
# o contexto, não uma fonte de fatos. Pedimos português, concisão e — crucial —
# recusa explícita quando a info não está no contexto (em vez de inventar).
_GENERATOR_SYSTEM_PROMPT = (
    "Você é um analista financeiro. Responda à PERGUNTA usando EXCLUSIVAMENTE as "
    "informações dos TRECHOS fornecidos. Regras: (1) não use conhecimento próprio "
    "nem suponha dados que não estejam nos trechos; (2) se os trechos não contêm a "
    "resposta, diga explicitamente que não há informação suficiente; (3) seja "
    "conciso e factual, citando números e períodos exatamente como aparecem nos "
    "trechos; (4) responda em português."
)


@lru_cache(maxsize=1)
def _get_answer_client() -> ChatOpenAI:
    """
    Client de geração, criado uma vez (lru_cache). Ponto de mock nos testes.

    Mesma estrutura do HyDE, mas temperature=0.0: resposta ancorada e
    reprodutível. with_structured_output NÃO é usado aqui de propósito — a saída
    é texto livre (uma resposta em prosa), não um schema; não há nada para
    parsear. (Contraste com o reranker, cuja saída É estruturada.)
    """
    return ChatOpenAI(
        api_key=SecretStr(settings.openai_api_key),
        model=settings.active_llm_model,
        temperature=0.0,
        max_retries=4,
        timeout=60.0,
    )


def _format_contexts(contexts: list[str]) -> str:
    """Numera os trechos no prompt — mesma ideia do reranker, leitura clara para o LLM."""
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))


async def generate_answer(question: str, contexts: list[str]) -> str:
    """
    Gera uma resposta à `question` ancorada em `contexts` (conteúdos dos chunks).

    Recebe `list[str]` (e não `list[RetrievedChunk]`) DE PROPÓSITO: o generator é
    agnóstico à origem dos trechos — vieram do baseline, do HyDE ou do re-rank,
    tanto faz. Quem orquestra (o runner do Passo 4) extrai os `.content` e passa
    aqui. Mesma filosofia de `search_by_embedding`/`rerank`: primitiva que não se
    importa com a procedência.

    Returns:
        A resposta em texto. Se `contexts` estiver vazio, devolve uma recusa
        determinística SEM chamar o LLM (recall zero -> admite que não sabe).
    """
    # Guard: sem contexto, não há o que ancorar. Recusa explícita e barata —
    # nunca pedimos ao LLM para "responder do nada" (seria convite à alucinação).
    if not contexts:
        return _NO_CONTEXT_ANSWER

    messages = [
        SystemMessage(content=_GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"PERGUNTA:\n{question}\n\nTRECHOS:\n{_format_contexts(contexts)}"),
    ]

    client = _get_answer_client()
    response = await client.ainvoke(messages)
    # response.content pode ser str ou list (multimodal). Normalizamos para str —
    # mesma defesa do HyDE; as métricas a jusante esperam texto puro.
    content = response.content
    answer = content if isinstance(content, str) else str(content)
    logger.debug("Generator produziu resposta de %d chars para %r", len(answer), question)
    return answer
