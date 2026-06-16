"""
Configuração compartilhada de testes.

O engine async é criado no nível de módulo em `finsight.db.session` com um pool
de conexões. Todos os testes async rodam num único event loop de sessão
(configurado em pyproject: asyncio_default_*_loop_scope = "session").

Aqui dispomos o engine ao FIM da sessão de testes — ainda dentro desse loop
vivo. Sem isso, o pool seria finalizado pelo garbage collector após o loop
fechar, e o asyncpg tentaria encerrar conexões num loop morto
("Event loop is closed", típico no Windows/ProactorEventLoop).
"""

from collections.abc import AsyncGenerator

import pytest_asyncio

from finsight.db.session import engine


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _dispose_engine() -> AsyncGenerator[None, None]:
    """Fecha o pool de conexões async ao término da sessão, dentro do loop ativo."""
    yield
    await engine.dispose()
