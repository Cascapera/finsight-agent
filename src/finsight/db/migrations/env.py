"""
Alembic environment — configura como as migrations são executadas.

Reescrito para suportar AsyncEngine (asyncpg).
O env.py padrão do `alembic init` usa conexão síncrona — incompatível com asyncpg.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Importamos Base para que o autogenerate detecte os modelos.
# A importação de models garante que as classes sejam registradas no metadata.
from finsight.db.models import Base  # noqa: F401 — importação necessária para registro
from finsight.db.session import settings

# config: objeto que lê o alembic.ini
config = context.config

# Sobrescreve sqlalchemy.url com o valor real das Settings
# — ignora o placeholder do alembic.ini
# Usamos database_url_sync porque o Alembic precisa de driver síncrono (psycopg2)
config.set_main_option("sqlalchemy.url", settings.database_url_sync)

# Configura logging conforme definido no alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# target_metadata: o Alembic compara este metadata com o schema atual do banco
# para gerar as migrations via --autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Modo offline: gera SQL sem se conectar ao banco.

    Útil para revisar o SQL antes de aplicar, ou para ambientes
    onde o banco não está acessível (ex: geração de scripts de CI).
    Uso: alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # compare_type=True: detecta mudanças de tipo de coluna no autogenerate
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Executa as migrations numa conexão já estabelecida."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # render_as_batch=False: não precisamos de batch mode (é para SQLite)
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Modo online com AsyncEngine.

    Cria um engine temporário (NullPool — sem pool para migrations),
    obtém uma conexão síncrona via run_sync, e executa as migrations.

    Por que run_sync? O Alembic internamente usa a API síncrona de Connection.
    AsyncConnection.run_sync() adapta essa API síncrona para o contexto async.
    """
    # async_engine_from_config: cria AsyncEngine a partir das configs do alembic.ini
    # poolclass=NullPool: sem pool — cada migration abre e fecha conexão individualmente
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        # run_sync: executa a função síncrona `do_run_migrations` dentro do contexto async
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point para o modo online — chamado pelo Alembic automaticamente."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
