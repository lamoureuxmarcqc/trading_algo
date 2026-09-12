from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.schemas.ai import AiBacktestRequest, AiPaperStageRequest, AiSizingRequest  # noqa: E402
from app.services.ai_trading_service import AiTradingService, PaperTradingDisabledError  # noqa: E402
from app.services.trading_algo_terminal import TradingAlgoTerminalService  # noqa: E402


class FakeExtractor:
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=260, freq="D")
        base = 100 if symbol == "AAPL" else 80
        step = 0.35 if symbol == "AAPL" else 0.12
        close = pd.Series([base + step * idx for idx in range(len(dates))], index=dates)
        return pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000,
            },
            index=dates,
        )

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        frame["SMA_20"] = frame["Close"].rolling(20, min_periods=1).mean()
        frame["SMA_50"] = frame["Close"].rolling(50, min_periods=1).mean()
        frame["RSI"] = 55.0
        frame["Volatility_20d"] = 0.18
        return frame


def _service() -> AiTradingService:
    terminal = TradingAlgoTerminalService(extractor=FakeExtractor())
    return AiTradingService(terminal)


def test_ai_service_exposes_diagnostics_explanations_and_prediction_bands() -> None:
    service = _service()

    diagnostics = service.get_diagnostics("AAPL")
    explanation = service.get_explanation("AAPL")
    bands = service.get_prediction_bands("AAPL")

    assert diagnostics.symbol == "AAPL"
    assert diagnostics.confidence_score > 0.5
    assert diagnostics.staleness.model_stale is False
    assert explanation.source == "heuristic_fallback"
    assert explanation.fallback_labeled is True
    assert explanation.features
    assert [band.horizon_days for band in bands.bands] == [1, 5, 30]
    assert bands.bands[0].probability_of_target is not None


def test_ai_service_sizes_positions_and_runs_backtests() -> None:
    service = _service()

    sizing = service.size_position(AiSizingRequest(symbol="AAPL", nav=500_000, max_position_weight=0.05))
    backtest = service.run_backtest(
        AiBacktestRequest(
            symbols=["AAPL", "MSFT"],
            lookback="90d",
            buy_probability_threshold=0.0,
            max_position_weight=0.05,
        )
    )

    assert sizing.sizing.recommended_notional <= 25_000
    assert sizing.sizing.recommended_quantity is not None
    assert sizing.sizing.risk_budget_consumed is not None
    assert len(backtest.results) == 2
    assert backtest.errors == []
    assert backtest.results[0].summary.lookback_days == 90


def test_ai_service_requires_explicit_paper_trading_enablement() -> None:
    service = _service()

    with pytest.raises(PaperTradingDisabledError):
        service.plan_paper_stage(AiPaperStageRequest(symbols=["AAPL"]))


def test_ai_service_plans_guardrailed_paper_order_candidates() -> None:
    service = _service()

    candidates, skipped, warnings = service.plan_paper_stage(
        AiPaperStageRequest(
            symbols=["AAPL"],
            buy_probability_threshold=0.0,
            paper_trading_enabled=True,
            max_notional=10_000,
            stop_loss=0.08,
            take_profit=0.18,
        )
    )

    assert skipped == []
    assert warnings == []
    assert candidates[0]["symbol"] == "AAPL"
    assert candidates[0]["side"] == "BUY"
    assert candidates[0]["quantity"] > 0
    assert candidates[0]["stop_price"] < candidates[0]["limit_price"]
    assert candidates[0]["take_profit_price"] > candidates[0]["limit_price"]
