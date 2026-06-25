"""
Schemas da API — contratos de entrada/saída HTTP, validados na borda.

Separados dos modelos do grafo (graph/state.py) DE PROPÓSITO: aqueles são o estado
interno do orquestrador; estes são o contrato público da API. Misturar os dois
acopla o formato do fio à mecânica interna — mudanças no state vazariam para os
clientes. A tradução AgentState/AnalysisEvent -> JSON da API mora aqui e em routes.
"""

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    """Corpo do POST /analyze. Valida e normaliza a entrada do usuário."""

    query: str = Field(min_length=1, description="Pergunta do usuário em linguagem natural.")
    ticker: str = Field(min_length=1, description="Código do ativo, ex: PETR4.")
    asset_type: str = Field(default="stock", description="Tipo do ativo: stock, fii, etf.")

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        """Tickers são maiúsculos (PETR4, não petr4) — normaliza para casar com os dados."""
        return v.strip().upper()

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        """Remove espaços nas pontas; min_length já garante não-vazio após o strip."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("query não pode ser vazia")
        return stripped
