"""
Configuração do banco de dados: Settings, engine async e session factory.

Único ponto de configuração da aplicação — nenhum outro módulo lê os.environ.
"""

from collections.abc import AsyncGenerator
from typing import Annotated, Any

from pydantic import Field, field_validator
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
# Normalização de URL de provedores gerenciados (Supabase/Upstash) — Semana 8
# ---------------------------------------------------------------------------
# Provedores entregam UMA URL `postgresql://...` sem o sufixo de driver e exigem TLS.
# O SQLAlchemy roteia o driver pelo scheme, então precisamos injetar `+asyncpg`/
# `+psycopg2`; e cada driver fala um dialeto de SSL diferente na query string:
#   - asyncpg  → token `ssl=require`   (o dialeto asyncpg do SQLAlchemy o repassa a
#                asyncpg.connect(ssl=...); 'require' = cifra sem verificar CA)
#   - psycopg2 → token `sslmode=require` (libpq)
# Funções puras (sem rede) → testáveis isoladamente.


def _inject_driver_and_ssl(raw_url: str, *, driver: str, ssl_token: str) -> str:
    """
    Reescreve uma URL `postgres(ql)://...` de provedor para `postgresql+<driver>://`
    e garante TLS via `<ssl_token>=require` se a URL ainda não traz config de SSL.
    """
    url = raw_url
    # Normaliza o scheme: cobre tanto `postgresql://` quanto o `postgres://` (Heroku-
    # style). count=1 garante que só o scheme é trocado, nunca algo no user/senha.
    for scheme in ("postgresql://", "postgres://"):
        if url.startswith(scheme):
            url = f"postgresql+{driver}://" + url[len(scheme) :]
            break
    # Só adiciona SSL se a URL não declarou nada — respeita uma config explícita do
    # operador (ex.: sslmode=verify-full com CA própria).
    if "ssl=" not in url and "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{ssl_token}=require"
    return url


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

    # --- Overrides de URL para provedores gerenciados (Supabase/Upstash) — Semana 8 ---
    # Quando setados (via `fly secrets set` no deploy), VENCEM as partes POSTGRES_*/
    # redis_* abaixo. Provedores gerenciados dão UMA URL única e exigem TLS — montar por
    # partes não cobriria SSL. Vazio (default) = dev local usa as partes do compose.
    # validation_alias: o env var é `DATABASE_URL`/`REDIS_URL` cru (o que o provedor
    # mostra), não `DATABASE_URL_OVERRIDE` — o nome do campo é só interno.
    database_url_override: str = Field(default="", validation_alias="DATABASE_URL")
    redis_url_override: str = Field(default="", validation_alias="REDIS_URL")

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

    # --- Eval (thresholds de aprovação das métricas próprias; ver evals/metrics.py) ---
    # Renomeado de ragas_* na Semana 4: construímos as métricas nós mesmos, RAGAS-lib
    # saiu da stack. Nomes alinhados às métricas reais (answer_relevancy com 'y').
    eval_min_faithfulness: float = 0.75
    eval_min_answer_relevancy: float = 0.70
    eval_min_context_recall: float = 0.65

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

        Se DATABASE_URL veio como secret (Supabase em prod), ele vence: normalizamos
        scheme + TLS. Senão, montamos a partir das partes POSTGRES_* (dev local).
        """
        if self.database_url_override:
            return _inject_driver_and_ssl(
                self.database_url_override, driver="asyncpg", ssl_token="ssl"
            )
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

        Mesmo override do async: em prod o Alembic (release_command no Fly) usa a
        DATABASE_URL do Supabase, com `sslmode=require` que o libpq/psycopg2 entende.
        """
        if self.database_url_override:
            return _inject_driver_and_ssl(
                self.database_url_override, driver="psycopg2", ssl_token="sslmode"
            )
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        # REDIS_URL como secret (Upstash em prod) vence: já vem como `rediss://` com TLS
        # e credenciais embutidas — o redis-py entende direto, sem normalização.
        if self.redis_url_override:
            return self.redis_url_override
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
