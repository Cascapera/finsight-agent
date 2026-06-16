"""
Alembic environment — configura como as migrations são executadas.

Migrations rodam de forma SÍNCRONA (psycopg2), que é o padrão recomendado do
Alembic — mais simples e sem o overhead de adaptar a API síncrona do Alembic
para async. O app em runtime continua 100% async (asyncpg) via session.py;
apenas as migrations usam o driver síncrono.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection

# Importamos Base para que o autogenerate detecte os modelos.
# A importação de models garante que as classes sejam registradas no metadata.
# A importação de Base também registra os modelos no metadata (usado em
# target_metadata abaixo para o autogenerate).
from finsight.db.models import Base
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


def run_migrations_online() -> None:
    """
    Modo online com Engine síncrono (psycopg2).

    Cria um Engine temporário a partir das configs do alembic.ini (onde
    sqlalchemy.url já foi sobrescrita por database_url_sync acima), abre uma
    conexão e executa as migrations diretamente — sem corrotina.

    poolclass=NullPool: sem pool — migrations abrem e fecham conexão individualmente.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
