"""
Tracing LangSmith — ponte entre o Settings do projeto e o LangChain.

O LangChain ativa o tracing lendo VARIÁVEIS DE AMBIENTE (LANGCHAIN_TRACING_V2,
LANGCHAIN_API_KEY, LANGCHAIN_PROJECT). Nosso `settings` já carrega esses valores do
.env, mas como é um objeto Python o LangChain não os enxerga. `configure_tracing`
faz a ponte: exporta as env vars quando o tracing está habilitado.

Por que env vars e não passar o client explicitamente? Porque o tracing do LangChain
é AUTOMÁTICO e global — uma vez ligado via ambiente, toda chamada de ChatOpenAI/
Runnable (HyDE, reranker, research, risk) é tracejada sem instrumentar cada uma. A
correlação com o execution_id vem do `config` passado no ainvoke (ver orchestrator).
"""

import logging
import os

from finsight.db.session import settings

logger = logging.getLogger(__name__)


def configure_tracing() -> bool:
    """
    Liga o tracing LangSmith via env vars, se habilitado em settings.

    Idempotente (só escreve em os.environ) e seguro de chamar no startup. Retorna True
    se o tracing foi ativado, False caso contrário — útil para log e para os testes.

    Não falha se a API key estiver vazia: o LangChain apenas não enviará os traces.
    Usamos os nomes LANGCHAIN_* (linha clássica, ainda suportada na v1) por casarem
    1:1 com os campos do Settings.
    """
    if not settings.langchain_tracing_v2:
        logger.debug("LangSmith tracing desabilitado (langchain_tracing_v2=False)")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

    logger.info("LangSmith tracing habilitado (project=%s)", settings.langchain_project)
    return True
