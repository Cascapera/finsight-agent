"""
RAG Agent — recupera trechos de relatórios PDF relevantes à pergunta.

É o terceiro ramo do fan-out (Semana 6), ao lado de Financial e Research. Responde
"o que os RELATÓRIOS dizem?" — busca por similaridade no pgvector sobre os PDFs
ingeridos, com over-fetch + re-ranking listwise (a pilha da Semana 3).

>>> O agente mais FINO dos quatro — e por quê <<<
Os outros nós têm um núcleo próprio (matemática no Financial, prompt de sentimento
no Research, síntese no Risk). Este não precisa: o trabalho pesado já existe na
camada de retrieval (Semana 3), desenhada para ser componível:

    retrieve_and_rerank(query, ticker=...)   over-fetch 20 -> rerank LLM -> top 5
    to_rag_output(chunks) -> RAGOutput        achata para o formato do state

Então `rag.py` é só a CASCA DE NÓ que adapta essa camada ao contrato do grafo. A
separação "infra PROPAGA exceção / nó CAPTURA" escrita no docstring do retriever.py
desde a Semana 3 se paga aqui: o retrieval levanta, e este nó é a fronteira que
traduz para `state["errors"]` — a regra de erro do projeto.
"""

import logging
from typing import Any

from finsight.graph.state import AgentState, RAGOutput
from finsight.retrieval.reranker import retrieve_and_rerank
from finsight.retrieval.retriever import to_rag_output

logger = logging.getLogger(__name__)

# Estratégia de retrieval do agente. Usamos over-fetch + re-ranking listwise
# (retrieve_and_rerank) como default por ser a mais robusta. `use_hyde=False` por
# ora: HyDE adiciona latência/custo (uma geração extra de LLM por consulta), e a
# escolha formal entre baseline/HyDE/rerank deveria sair do eval suite (Semana 4)
# rodado sobre os PDFs reais. Constante para tornar a troca um diff de uma linha.
_USE_HYDE = False


async def rag_node(state: AgentState) -> dict[str, Any]:
    """
    Nó do LangGraph: recupera os trechos de relatórios relevantes -> {"rag": RAGOutput}.

    Roda no fan-out, em paralelo com financial_node e research_node. Em produção usa
    `session=None` -> a camada de retrieval abre e fecha a própria sessão (padrão
    "owned"); o nó não gerencia transação.

    Fronteira de erro: o retrieval PROPAGA (banco fora do ar, falha no embedding ou
    no LLM do re-rank). Capturamos aqui e devolvemos, além do RAGOutput vazio,
    `{"errors": [...]}` — degradação graciosa. RAGOutput vazio = "sem trechos", que o
    synthesize do Risk já serializa como "indisponíveis". Nunca propagamos.
    """
    ticker = state["ticker"]
    query = state["query"]
    try:
        chunks = await retrieve_and_rerank(query, ticker=ticker, use_hyde=_USE_HYDE)
    except Exception as exc:  # fronteira do nó: captura tudo, nunca propaga
        logger.warning("rag_node: falha ao recuperar trechos de %s: %s", ticker, exc)
        return {
            "rag": RAGOutput(),
            "errors": [f"rag: falha ao recuperar trechos de {ticker}: {exc}"],
        }

    rag = to_rag_output(chunks)
    logger.debug("rag_node: %s -> %d trecho(s) recuperado(s)", ticker, len(rag.chunks))
    return {"rag": rag}
