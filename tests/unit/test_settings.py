"""
Testes da normalização de URL para provedores gerenciados (Semana 8 — deploy).

Foco: garantir que DATABASE_URL/REDIS_URL (Supabase/Upstash) sejam reescritos com o
driver correto e TLS, sem depender de rede. Tudo aqui é string-in/string-out.
"""

from finsight.db.session import Settings, _inject_driver_and_ssl

# ---------------------------------------------------------------------------
# Helper puro
# ---------------------------------------------------------------------------


def test_inject_async_driver_and_ssl() -> None:
    raw = "postgresql://user:pass@db.supabase.co:5432/postgres"
    out = _inject_driver_and_ssl(raw, driver="asyncpg", ssl_token="ssl")
    assert out == "postgresql+asyncpg://user:pass@db.supabase.co:5432/postgres?ssl=require"


def test_inject_sync_driver_and_ssl() -> None:
    raw = "postgresql://user:pass@db.supabase.co:5432/postgres"
    out = _inject_driver_and_ssl(raw, driver="psycopg2", ssl_token="sslmode")
    assert out == "postgresql+psycopg2://user:pass@db.supabase.co:5432/postgres?sslmode=require"


def test_inject_handles_postgres_scheme_alias() -> None:
    # Alguns provedores (Heroku-style) usam `postgres://` em vez de `postgresql://`.
    raw = "postgres://u:p@h:5432/d"
    out = _inject_driver_and_ssl(raw, driver="asyncpg", ssl_token="ssl")
    assert out.startswith("postgresql+asyncpg://u:p@h:5432/d")
    assert out.endswith("?ssl=require")


def test_inject_preserves_existing_ssl_config() -> None:
    # Se o operador já declarou SSL, NÃO duplicamos nem sobrescrevemos.
    raw = "postgresql://u:p@h:5432/d?sslmode=verify-full"
    out = _inject_driver_and_ssl(raw, driver="psycopg2", ssl_token="sslmode")
    assert out == "postgresql+psycopg2://u:p@h:5432/d?sslmode=verify-full"


def test_inject_appends_ssl_with_ampersand_when_query_exists() -> None:
    raw = "postgresql://u:p@h:5432/d?application_name=finsight"
    out = _inject_driver_and_ssl(raw, driver="asyncpg", ssl_token="ssl")
    assert out == "postgresql+asyncpg://u:p@h:5432/d?application_name=finsight&ssl=require"


# ---------------------------------------------------------------------------
# Properties do Settings — override vence as partes; vazio cai no fallback local
# ---------------------------------------------------------------------------


def _settings(**overrides: str) -> Settings:
    """Settings determinístico: ignora .env do disco, injeta só o necessário."""
    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        openai_api_key="sk-test",
        tavily_api_key="tvly-test",
    )
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


def test_database_url_uses_override_when_present() -> None:
    s = _settings(database_url_override="postgresql://u:p@host:6543/db")
    assert s.database_url == "postgresql+asyncpg://u:p@host:6543/db?ssl=require"
    assert s.database_url_sync == "postgresql+psycopg2://u:p@host:6543/db?sslmode=require"


def test_database_url_falls_back_to_parts_without_override() -> None:
    s = _settings()  # sem override → monta das partes (dev local), sem SSL
    assert s.database_url == ("postgresql+asyncpg://finsight:finsight@localhost:5432/finsight")
    assert "ssl" not in s.database_url


def test_redis_url_uses_override_verbatim() -> None:
    s = _settings(redis_url_override="rediss://default:tok@host.upstash.io:6379")
    assert s.redis_url == "rediss://default:tok@host.upstash.io:6379"


def test_redis_url_falls_back_to_parts_without_override() -> None:
    s = _settings()
    assert s.redis_url == "redis://localhost:6379/0"
