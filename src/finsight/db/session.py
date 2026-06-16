"""
Configuração do banco de dados: Settings, engine async e session factory.

Único ponto de configuração da aplicação — nenhum outro módulo lê os.environ.
"""

from collections.abc import AsyncGenerator
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

# ---------------------------------------------------------------------------
# Settings — lê .env e valida tipos na inicialização
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Configuração centralizada da aplicação via variáveis de ambiente.

    pydantic-settings lê do .env automaticamente. Se uma variável obrigatória
    (sem default) estiver ausente, ValidationError é levantado na importação
    do módulo — fail fast, não em runtime durante uma request.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # case_sensitive=False: POSTGRES_HOST e postgres_host são equivalentes
        case_sensitive=False,
        # extra="ignore": variáveis no .env que não estão no modelo são ignoradas
        # sem isso, qualquer variável extra levantaria ValidationError
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str
    openai_model_dev: str = "gpt-4o-mini"
    openai_model_prod: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_cost_per_query_usd: float = 0.10

    # --- Tavily ---
    tavily_api_key: str

    # --- LangSmith ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "finsight-agent-dev"

    # --- PostgreSQL ---
    postgres_user: str = "finsight"
    postgres_password: str = "finsight"
    postgres_db: str = "finsight"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_pool_min_size: int = 2
    postgres_pool_max_size: int = 10

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_cache_ttl_seconds: int = 3600

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"
    secret_key: str = "dev-insecure-secret-change-in-production"
    cors_origins: list[str] = ["http://localhost:3000"]

    # --- Ingestion ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    embedding_batch_size: int = 100

    # --- Prometheus ---
    prometheus_enabled: bool = True
    prometheus_metrics_path: str = "/metrics"

    # --- RAGAS ---
    ragas_min_faithfulness: float = 0.75
    ragas_min_answer_relevance: float = 0.70
    ragas_min_context_recall: float = 0.65

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Aceita tanto lista quanto string separada por vírgula do .env."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def active_llm_model(self) -> str:
        """Retorna o modelo correto baseado no ambiente — sem if/else espalhado no código."""
        return self.openai_model_prod if self.is_production else self.openai_model_dev

    @property
    def database_url(self) -> str:
        """
        Monta a URL do PostgreSQL para o driver asyncpg.

        asyncpg usa o scheme `postgresql+asyncpg://` — diferente do psycopg2
        que usa `postgresql://`. O SQLAlchemy roteia para o driver correto
        baseado no scheme.
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """
        URL síncrona para o Alembic.

        Alembic roda migrations de forma síncrona — precisa de psycopg2 ou
        do driver síncrono. Usamos `postgresql+psycopg2` aqui.
        Nota: requer `psycopg2-binary` instalado (adicionamos nas deps do Alembic).
        """
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Instância global — importada pelos outros módulos
# Criada uma vez no startup, não a cada request
# Nota: os campos obrigatórios (openai_api_key, tavily_api_key) vêm do .env em
# runtime; o mypy não enxerga isso e exigiria passá-los aqui, daí o ignore abaixo.
settings = Settings()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Engine + Session factory
# ---------------------------------------------------------------------------


def create_engine(*, for_alembic: bool = False) -> AsyncEngine:
    """
    Cria o AsyncEngine do SQLAlchemy.

    Args:
        for_alembic: quando True, usa NullPool — desabilita connection pooling
                     para que o Alembic não tente reusar conexões async em
                     contexto síncrono (causa deadlock).
    """
    kwargs: dict[str, Any] = {
        "echo": not settings.is_production,  # loga SQL em dev, silencia em prod
    }

    if for_alembic:
        # NullPool: sem pool — cada operação abre e fecha conexão individualmente
        # necessário para o env.py do Alembic que roda em contexto síncrono
        kwargs["poolclass"] = NullPool
    else:
        # AsyncAdaptedQueuePool (default async): mantém pool de conexões abertas
        # pool_size = min_size inicial; max_overflow = conexões extras sob carga
        kwargs["pool_size"] = settings.postgres_pool_min_size
        kwargs["max_overflow"] = settings.postgres_pool_max_size - settings.postgres_pool_min_size
        # pool_pre_ping: antes de usar uma conexão do pool, testa se ainda está viva
        # evita "SSL connection has been closed unexpectedly" após idle timeout
        kwargs["pool_pre_ping"] = True

    return create_async_engine(settings.database_url, **kwargs)


# Engine da aplicação — criado uma vez, reutilizado por todas as sessions
engine = create_engine()

# async_sessionmaker: factory de AsyncSession
# expire_on_commit=False: após commit, os atributos dos objetos não expiram
# sem isso, acessar obj.id após commit levanta uma query adicional (lazy load)
# que falha em contexto async por não ter sessão ativa
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Base declarativa para os modelos ORM
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    Base para todos os modelos SQLAlchemy do projeto.

    Importada em cada arquivo de modelo e pelo Alembic para autogenerate
    de migrations (alembic revision --autogenerate detecta mudanças nos
    modelos que herdam desta Base).
    """

    pass


# ---------------------------------------------------------------------------
# Dependency do FastAPI
# ---------------------------------------------------------------------------


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency do FastAPI que injeta uma AsyncSession por request.

    Uso nos endpoints:
        async def my_endpoint(session: Annotated[AsyncSession, Depends(get_session)]):
            ...

    O `yield` divide a função em dois momentos:
    - Antes do yield: abre a sessão e a injeta no endpoint
    - Depois do yield (finally): garante commit ou rollback e fecha a sessão
      independente de exception — sem vazamento de conexão
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        # `async with AsyncSessionLocal()` já chama session.close() ao sair
        # não precisamos fechar manualmente


# Tipo anotado para uso com Depends — evita repetição em cada endpoint
SessionDep = Annotated[AsyncSession, None]  # substituído por Depends em routes.py
