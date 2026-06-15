from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.schemas.terminal import TradingAlgoCommandRequest  # noqa: E402
from app.services.trading_algo_terminal import TradingAlgoTerminalService  # noqa: E402


class FakeExtractor:
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=260, freq="D")
        base = 100 if symbol == "AAPL" else 80
        step = 0.35 if symbol == "AAPL" else -0.05
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


def test_trading_algo_terminal_analyzes_symbols_without_shelling_out() -> None:
    service = TradingAlgoTerminalService(extractor=FakeExtractor())

    response = service.run(
        TradingAlgoCommandRequest(
            command="analyze",
            symbols=["AAPL", "MSFT"],
            period="1y",
        )
    )

    assert response.status == "ok"
    assert [item.symbol for item in response.analyses] == ["AAPL", "MSFT"]
    assert response.analyses[0].trend == "bullish"
    assert response.analyses[0].recommendation == "buy"


def test_trading_algo_terminal_rejects_unknown_command() -> None:
    service = TradingAlgoTerminalService(extractor=FakeExtractor())

    response = service.run(TradingAlgoCommandRequest(command="shell", symbols=["AAPL"]))

    assert response.status == "error"
    assert response.analyses == []
    assert "Allowed commands" in response.errors[0]
