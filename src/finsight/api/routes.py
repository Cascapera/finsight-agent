"""
Rotas da API — health check + o endpoint de análise com streaming SSE.

O coração é `POST /analyze`: dispara o grafo e devolve um stream Server-Sent Events,
um evento por nó concluído + um evento final. A tradução AnalysisEvent -> evento SSE
acontece aqui; o orquestrador permanece agnóstico a HTTP (ver orchestrator.py).
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from finsight.api.schemas import AnalyzeRequest
from finsight.graph.orchestrator import run_analysis_stream

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — usado pelo Fly.io (Semana 8) e por monitoração simples."""
    return {"status": "ok"}


async def _sse_events(request: AnalyzeRequest) -> AsyncIterator[dict[str, Any]]:
    """
    Adapta o stream de AnalysisEvent do orquestrador para o formato do sse-starlette.

    Cada AnalysisEvent vira um dict {"event": <tipo>, "data": <json>}: o sse-starlette
    serializa isso como `event: progress\\ndata: {...}\\n\\n` no fio. Usar `ev.type`
    como NOME do evento SSE permite ao cliente escutar "progress" e "complete"
    separadamente (addEventListener por tipo).
    """
    async for ev in run_analysis_stream(request.query, request.ticker, request.asset_type):
        yield {"event": ev.type, "data": ev.model_dump_json()}


@router.post("/analyze")
async def analyze(request: AnalyzeRequest) -> EventSourceResponse:
    """
    Dispara a análise multi-agente e faz streaming do progresso via SSE.

    POST (não GET): o body carrega query/ticker com validação. Clientes consomem com
    `fetch` + leitura do stream; o `EventSource` nativo do browser só faz GET — quem
    precisar dele usaria uma variante GET com query params (fora do escopo aqui).

    EventSourceResponse cuida dos headers (text/event-stream, no-cache) e de drenar o
    gerador até o fim, fechando a conexão quando o evento "complete" é emitido.
    """
    logger.debug("POST /analyze: ticker=%s query=%r", request.ticker, request.query)
    return EventSourceResponse(_sse_events(request))
