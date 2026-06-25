"""
Aplicação FastAPI — factory + instância exportada para o uvicorn.

Padrão factory (`create_app`) em vez de um `app` montado no nível do módulo com
side effects: a factory é testável (cada teste cria sua app limpa) e deixa explícito
o que é configuração (CORS, rotas) vs. instância. A instância `app` no fim é o que o
uvicorn carrega: `uvicorn finsight.api.app:app`.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finsight.api.routes import router
from finsight.db.session import settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Monta a aplicação: metadados, CORS e rotas. Sem efeito de rede ao criar."""
    app = FastAPI(
        title="FinSight Agent",
        description="Análise multi-agente de ativos financeiros (LangGraph + RAG).",
        version="0.1.0",
    )

    # CORS: o frontend (origens em settings.cors_origins) precisa poder chamar a API
    # do navegador. allow_credentials para cookies/autorização futura; métodos e
    # headers liberados para simplicidade no escopo atual.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    logger.debug("FastAPI app criada (cors_origins=%s)", settings.cors_origins)
    return app


# Instância carregada pelo uvicorn (uvicorn finsight.api.app:app). Criada uma vez no
# import — barato, sem rede. O grafo é compilado por requisição dentro do stream.
app = create_app()
