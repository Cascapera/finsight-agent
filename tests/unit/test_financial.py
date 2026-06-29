"""
Testes do Financial Agent (Semana 5, Passo 1) — sem rede.

Duas frentes:
  * compute_metrics: matemática pura. Fabricamos séries de preços com
    propriedades CONHECIDAS (retorno diário constante, série constante, série
    curta) e checamos os números — sem mock, é função pura.
  * financial_node: o contrato de nó. Mockamos `_fetch_prices` (o ponto de rede)
    para simular sucesso e falha, e verificamos o patch devolvido — incluindo a
    captura de erro em state["errors"] (o nó nunca propaga).
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from finsight.agents import financial
from finsight.agents.financial import (
    TRADING_DAYS,
    _to_yahoo_symbol,
    compute_metrics,
    financial_node,
)
from finsight.graph.state import FinancialOutput


def _make_state(ticker: str = "PETR4") -> Any:
    """AgentState mínimo para o nó — só `ticker` é lido pelo financial_node."""
    return {"ticker": ticker, "query": "analise", "asset_type": "stock"}


# ===========================================================================
# _to_yahoo_symbol — mapeamento de ticker -> símbolo do Yahoo (puro, sem rede)
# ===========================================================================


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("PETR4", "PETR4.SA"),  # ação PN da B3
        ("VALE3", "VALE3.SA"),  # ação ON da B3
        ("HGLG11", "HGLG11.SA"),  # FII (4 letras + 11)
        ("petr4", "PETR4.SA"),  # normaliza caixa antes de casar o padrão
        ("AAPL", "AAPL"),  # ticker US (sem dígito) -> intacto
        ("PETR4.SA", "PETR4.SA"),  # já sufixado -> não duplica
        ("BRK.B", "BRK.B"),  # ponto explícito de outra bolsa -> respeitado
    ],
)
def test_to_yahoo_symbol(ticker: str, expected: str) -> None:
    """Tickers da B3 ganham '.SA'; os demais passam inalterados."""
    assert _to_yahoo_symbol(ticker) == expected


# ===========================================================================
# compute_metrics — matemática pura
# ===========================================================================


def test_flat_price_series_has_undefined_sharpe() -> None:
    """
    Preços LITERALMENTE constantes: retornos exatamente 0, vol exatamente 0.
    -> Sharpe None (indefinido, evita divisão por zero), VaR 0, acumulado 0.
    """
    prices = pd.Series([100.0] * (TRADING_DAYS + 1))
    out = compute_metrics(prices)

    assert out.current_price == pytest.approx(100.0)
    assert out.sharpe_ratio is None  # vol == 0 exato -> indefinido
    assert out.volatility_annualized == pytest.approx(0.0)
    assert out.var_95 == pytest.approx(0.0)
    assert out.cumulative_return_1y == pytest.approx(0.0)


def test_steady_growth_cumulative_return() -> None:
    """
    Série que sobe 1% ao dia: retornos ~constantes (com ruído de float).
    - retorno acumulado dos últimos 252 retornos = (1.01)^252 - 1
    - vol ~0 (mas não exata) -> Sharpe é um número grande positivo, NÃO None.
    """
    n = TRADING_DAYS
    prices = pd.Series([100.0 * (1.01**i) for i in range(n + 1)])
    out = compute_metrics(prices)

    assert out.current_price == pytest.approx(100.0 * (1.01**n))
    assert out.cumulative_return_1y == pytest.approx(1.01**TRADING_DAYS - 1.0)
    # Risco ~nulo + retorno consistente -> Sharpe altíssimo (não indefinido aqui,
    # pois o desvio de float é minúsculo mas > 0). Documenta o caso-limite.
    assert out.sharpe_ratio is not None
    assert out.sharpe_ratio > 0


def test_sharpe_matches_manual_formula() -> None:
    """Sharpe anualizado bate com o cálculo manual sobre os mesmos retornos."""
    rng = np.random.default_rng(42)
    # Série de preços a partir de retornos aleatórios (com dispersão real).
    daily_returns = rng.normal(0.0005, 0.01, size=300)
    prices = pd.Series(100.0 * np.cumprod(1.0 + np.concatenate([[0.0], daily_returns])))
    out = compute_metrics(prices)

    realized = prices.pct_change().dropna()
    expected_sharpe = realized.mean() / realized.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert out.sharpe_ratio is not None
    assert out.sharpe_ratio == pytest.approx(expected_sharpe)
    assert out.volatility_annualized == pytest.approx(realized.std(ddof=1) * np.sqrt(TRADING_DAYS))


def test_var_95_is_positive_loss_magnitude() -> None:
    """VaR 95% é a magnitude positiva da perda no percentil 5 dos retornos."""
    rng = np.random.default_rng(7)
    daily_returns = rng.normal(0.0, 0.02, size=500)
    prices = pd.Series(100.0 * np.cumprod(1.0 + np.concatenate([[0.0], daily_returns])))
    out = compute_metrics(prices)

    realized = prices.pct_change().dropna()
    expected_var = -np.percentile(realized, 5)
    assert out.var_95 == pytest.approx(expected_var)
    assert out.var_95 > 0  # mercado com volatilidade -> perda no pior 5% é positiva


def test_risk_free_rate_lowers_sharpe() -> None:
    """rf > 0 desconta o excesso de retorno -> Sharpe menor que com rf=0."""
    rng = np.random.default_rng(1)
    daily_returns = rng.normal(0.001, 0.01, size=300)
    prices = pd.Series(100.0 * np.cumprod(1.0 + np.concatenate([[0.0], daily_returns])))

    sharpe_no_rf = compute_metrics(prices, risk_free_rate=0.0).sharpe_ratio
    sharpe_with_rf = compute_metrics(prices, risk_free_rate=0.10).sharpe_ratio
    assert sharpe_no_rf is not None and sharpe_with_rf is not None
    assert sharpe_with_rf < sharpe_no_rf


def test_short_series_reports_only_price() -> None:
    """Menos que o mínimo de observações: só o preço atual; risco fica None."""
    prices = pd.Series([100.0, 101.0, 102.0])  # 2 retornos < _MIN_OBS_FOR_METRICS
    out = compute_metrics(prices)
    assert out.current_price == pytest.approx(102.0)
    assert out.sharpe_ratio is None
    assert out.var_95 is None
    assert out.volatility_annualized is None


def test_empty_series_is_all_none() -> None:
    """Série vazia: output válido com tudo None (sem dados)."""
    out = compute_metrics(pd.Series(dtype="float64"))
    assert out.current_price is None
    assert out.sharpe_ratio is None


# ===========================================================================
# financial_node — contrato de nó
# ===========================================================================


@pytest.mark.asyncio
async def test_node_returns_financial_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sucesso: o nó devolve {"financial": FinancialOutput} e nenhum erro."""
    prices = pd.Series([100.0 * (1.0 + 0.001 * (i % 3 - 1)) ** i for i in range(60)])

    def fake_fetch(ticker: str) -> pd.Series:
        assert ticker == "PETR4"
        return prices

    monkeypatch.setattr(financial, "_fetch_prices", fake_fetch)

    patch = await financial_node(_make_state("PETR4"))

    assert "errors" not in patch
    assert isinstance(patch["financial"], FinancialOutput)
    assert patch["financial"].current_price is not None


@pytest.mark.asyncio
async def test_node_captures_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falha de rede: o nó NÃO propaga — captura e escreve em errors."""

    def boom(ticker: str) -> pd.Series:
        raise RuntimeError("yfinance offline")

    monkeypatch.setattr(financial, "_fetch_prices", boom)

    patch = await financial_node(_make_state("VALE3"))

    # Output vazio (degradação graciosa) + erro registrado para o reducer `add`.
    assert isinstance(patch["financial"], FinancialOutput)
    assert patch["financial"].sharpe_ratio is None
    assert len(patch["errors"]) == 1
    assert "VALE3" in patch["errors"][0]
