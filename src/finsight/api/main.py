"""
Ponto de entrada de PROCESSO da API — o módulo que o uvicorn e o console script carregam.

Por que existe um `main.py` separado do `app.py`?
- `app.py` tem a responsabilidade de COMO a aplicação é montada: a factory
  `create_app()` (tracing, CORS, rotas) e a instância `app`.
- `main.py` tem a responsabilidade de COMO o servidor sobe: é a borda de processo.

Essa separação dá um único módulo estável de entrada para todos os pontos de boot:
  - Dockerfile:        ENTRYPOINT ["uvicorn", "finsight.api.main:app"]
  - console script:    finsight-serve = "finsight.api.main:run"  (pyproject.toml)
  - uvicorn manual:    uvicorn finsight.api.main:app

Sem este arquivo, ambas as referências acima apontavam para um módulo inexistente —
o container subia e morria com ModuleNotFoundError (bug descoberto na Semana 8, nunca
pegou antes porque só rodávamos `uvicorn finsight.api.app:app` em dev).
"""

import uvicorn

from finsight.api.app import app
from finsight.db.session import settings

# Re-exporta `app` explicitamente: é o atributo que `uvicorn finsight.api.main:app`
# importa. `run` entra no __all__ por ser o alvo do console script `finsight-serve`.
__all__ = ["app", "run"]


def run() -> None:
    """
    Sobe o servidor uvicorn lendo host/porta dos Settings — usado por `finsight-serve`.

    Passamos o objeto `app` já importado (não a string "finsight.api.main:app"): como
    não há reload/workers aqui, não precisamos que o uvicorn reimporte o módulo num
    subprocesso. Em container, o boot real é o ENTRYPOINT do Dockerfile (uvicorn CLI
    com --host/--port/--workers), não esta função — `run` é a conveniência de CLI local.
    """
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
