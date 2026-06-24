"""
Financial Agent — métricas quantitativas de risco/retorno a partir do histórico.

É o primeiro nó do grafo (Semana 5) e o único DETERMINÍSTICO: não há LLM aqui,
só matemática financeira sobre a série de preços (Sharpe, VaR, retorno acumulado,
volatilidade). Lição que vale reter: nem todo "agente" precisa de um LLM — um nó
é qualquer função `(state) -> patch parcial do state`, e este faz contas.

>>> Duas camadas, deliberadamente separadas (mesmo padrão do retrieval) <<<

    compute_metrics(prices) -> FinancialOutput   # NÚCLEO puro: série -> métricas
    financial_node(state)   -> dict              # NÓ: busca dados, chama o núcleo,
                                                 #     captura erro, devolve patch

`compute_metrics` é pura e sem rede — testável com uma série fabricada e
resultado determinístico (é onde o rigor quant mora). `_fetch_prices` é o único
ponto que toca a rede (yfinance) e, por isso, o ponto de mock dos testes.

>>> O contrato de nó do LangGraph <<<
Um nó NÃO muta o state: devolve um dict parcial que o LangGraph mescla. Aqui é
`{"financial": FinancialOutput(...)}`. A regra de erro do projeto vive no nó:
capturamos a exceção e devolvemos `{"errors": [msg]}` — NUNCA propagamos. Como
`AgentState.errors` tem reducer `add`, erros de nós paralelos se acumulam.
Degradação graciosa: se os dados faltam, devolvemos um FinancialOutput com campos
None (o schema já permite) em vez de derrubar o grafo inteiro.
"""

import asyncio
import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from finsight.graph.state import AgentState, FinancialOutput

logger = logging.getLogger(__name__)

# Dias úteis de pregão por ano — fator de anualização padrão do mercado.
# Volatilidade diária * sqrt(252) -> anual; idem para o Sharpe.
TRADING_DAYS = 252

# Mínimo de retornos diários para as métricas de risco fazerem sentido. Abaixo
# disso (ex: ativo recém-listado) Sharpe/VaR são ruído estatístico — devolvemos
# None nesses campos (honesto) em vez de um número sem significado. O preço atual
# ainda é reportado quando há ao menos um ponto.
_MIN_OBS_FOR_METRICS = 20

# Janela do retorno acumulado: últimos 252 dias úteis (~1 ano), casando com o
# `period="1y"` da busca. Documentado no FinancialOutput.cumulative_return_1y.
_RETURN_WINDOW = TRADING_DAYS


def compute_metrics(prices: pd.Series, *, risk_free_rate: float = 0.0) -> FinancialOutput:
    """
    Calcula as métricas de risco/retorno a partir de uma série de preços de fechamento.

    Núcleo puro: entra uma série (preços ajustados, ordenados por data), saem as
    métricas. Sem rede, sem efeito colateral — determinístico e testável.

    Args:
        prices: série de preços de fechamento AJUSTADOS (dividendos/splits já
            incorporados), do mais antigo ao mais recente.
        risk_free_rate: taxa livre de risco ANUAL (ex: 0.105 p/ SELIC ~10,5%).
            Default 0.0 = Sharpe sobre o retorno bruto. Em produção, para ativos
            BR, viria do CDI/SELIC; mantemos parametrizável e fora do cálculo
            para o núcleo não depender de configuração externa.

    Convenções (não-óbvias, valem comentário):
      - Sharpe ANUALIZADO: (média_excesso_diário / desvio_diário) * sqrt(252).
      - VaR 95% HISTÓRICO diário, reportado como MAGNITUDE de perda positiva: é
        o simétrico do percentil 5 dos retornos. var_95=0.03 significa "no pior
        5% dos dias, a perda diária é >= 3%". Histórico (não-paramétrico): não
        assume normalidade, capta caudas gordas reais do ativo.
      - Volatilidade ANUALIZADA: desvio diário (amostral, ddof=1) * sqrt(252).
      - Retorno acumulado: produto de (1+r) sobre os últimos 252 retornos - 1.
    """
    prices = prices.dropna()

    # Sem dados suficientes nem para um retorno: devolve tudo None. FinancialOutput()
    # com defaults None é um output VÁLIDO — sinaliza "sem dados" sem quebrar o schema.
    if len(prices) < 2:
        return FinancialOutput(current_price=_last_price(prices))

    current_price = _last_price(prices)
    # pct_change: r_t = P_t / P_{t-1} - 1. dropna() remove o primeiro (NaN, sem
    # anterior). É a base de todas as métricas de risco.
    daily_returns = prices.pct_change().dropna()

    # Poucos retornos: o preço atual é confiável, as métricas de risco não.
    # Reportamos só o preço — None nas demais é o diagnóstico correto.
    if len(daily_returns) < _MIN_OBS_FOR_METRICS:
        return FinancialOutput(current_price=current_price)

    mean_daily = float(daily_returns.mean())
    # ddof=1: desvio AMOSTRAL (n-1) — estimamos a vol de uma amostra, não da
    # população. Convenção em finanças; difere de ddof=0 (numpy default) em
    # amostras pequenas.
    std_daily = float(daily_returns.std(ddof=1))

    volatility_annualized = std_daily * np.sqrt(TRADING_DAYS)

    # Sharpe indefinido se a vol é zero (série constante) — evita divisão por zero
    # e reporta None em vez de inf/NaN.
    rf_daily = risk_free_rate / TRADING_DAYS
    sharpe_ratio = (
        (mean_daily - rf_daily) / std_daily * np.sqrt(TRADING_DAYS) if std_daily > 0 else None
    )

    # VaR histórico: percentil 5 dos retornos. Negamos para reportar perda como
    # número positivo. np.percentile interpola entre observações (default linear).
    var_95 = float(-np.percentile(daily_returns, 5))

    # Retorno acumulado dos últimos _RETURN_WINDOW retornos. .tail garante a
    # janela mesmo com mais de 1 ano de dados; com menos, usa o que houver.
    window = daily_returns.tail(_RETURN_WINDOW)
    cumulative_return_1y = float((1.0 + window).prod() - 1.0)

    return FinancialOutput(
        sharpe_ratio=sharpe_ratio,
        var_95=var_95,
        cumulative_return_1y=cumulative_return_1y,
        volatility_annualized=volatility_annualized,
        current_price=current_price,
    )


def _last_price(prices: pd.Series) -> float | None:
    """Último preço da série, ou None se vazia. Centraliza o guard de série vazia."""
    if prices.empty:
        return None
    return float(prices.iloc[-1])


def _fetch_prices(ticker: str) -> pd.Series:
    """
    Busca o histórico de 1 ano de fechamentos AJUSTADOS via yfinance.

    Síncrono e bloqueante (yfinance faz HTTP) — por isso o nó o chama via
    `asyncio.to_thread`, para não travar o event loop. É o ÚNICO ponto de rede e,
    portanto, o ponto de mock: os testes substituem esta função por uma série
    fabricada, sem tocar a internet.

    auto_adjust=True: usa preços ajustados por proventos/splits — obrigatório para
    o cálculo de retorno não ter saltos artificiais em datas de dividendo/split.
    """
    history = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
    # Ticker inexistente devolve DataFrame vazio (sem coluna "Close" garantida);
    # tratamos como série vazia -> compute_metrics devolve output "sem dados".
    if history.empty or "Close" not in history.columns:
        return pd.Series(dtype="float64")
    close = history["Close"].dropna()
    return close


async def financial_node(state: AgentState) -> dict[str, Any]:
    """
    Nó do LangGraph: busca o histórico do `ticker` do state e calcula as métricas.

    Devolve um patch parcial `{"financial": FinancialOutput}`. Em falha de rede
    (yfinance fora do ar, timeout) captura a exceção e devolve, ALÉM do output
    vazio, `{"errors": [...]}` — degradação graciosa. O grafo segue: a síntese
    final usa o que os outros agentes trouxeram, com uma nota de erro registrada.
    """
    ticker = state["ticker"]
    try:
        # to_thread: roda a chamada bloqueante do yfinance num worker thread,
        # liberando o event loop para os outros nós do fan-out (ex: Research).
        prices = await asyncio.to_thread(_fetch_prices, ticker)
    except Exception as exc:  # fronteira do nó: capturamos tudo, nunca propagamos
        logger.warning("financial_node: falha ao buscar %s: %s", ticker, exc)
        return {
            "financial": FinancialOutput(),
            "errors": [f"financial: falha ao buscar dados de {ticker}: {exc}"],
        }

    metrics = compute_metrics(prices)
    logger.debug(
        "financial_node: %s -> sharpe=%s var95=%s ret1y=%s",
        ticker,
        metrics.sharpe_ratio,
        metrics.var_95,
        metrics.cumulative_return_1y,
    )
    return {"financial": metrics}
