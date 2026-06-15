from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Iterable

import pandas as pd

from app.schemas.terminal import (
    TradingAlgoCommandRequest,
    TradingAlgoCommandResponse,
    TradingAlgoSymbolAnalysis,
)


class TradingAlgoTerminalService:
    """Small API-safe bridge from the institutional terminal to trading_algo."""

    DEFAULT_SCREEN_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "JPM", "XOM", "LLY"]
    ALLOWED_COMMANDS = {"analyze", "compare", "screen"}

    def __init__(
        self,
        extractor: Any | None = None,
        risk_manager: Any | None = None,
    ) -> None:
        if extractor is None:
            from trading_algo.data.data_extraction import StockDataExtractor

            extractor = StockDataExtractor()
        if risk_manager is None:
            from trading_algo.risk.risk_manager import RiskManager

            risk_manager = RiskManager()

        self.extractor = extractor
        self.risk_manager = risk_manager

    def run(self, payload: TradingAlgoCommandRequest) -> TradingAlgoCommandResponse:
        command = payload.command.strip().lower()
        if command not in self.ALLOWED_COMMANDS:
            return self._response(
                command=command or "unknown",
                status="error",
                summary=f"Unsupported trading-algo command: {payload.command}",
                analyses=[],
                errors=[f"Allowed commands: {', '.join(sorted(self.ALLOWED_COMMANDS))}"],
            )

        symbols = self._symbols_for(payload)
        analyses: list[TradingAlgoSymbolAnalysis] = []
        errors: list[str] = []

        for symbol in symbols:
            try:
                analyses.append(self._analyze_symbol(symbol, payload.period))
            except Exception as exc:  # pragma: no cover - defensive guard for data providers
                errors.append(f"{symbol}: {exc}")

        if command in {"compare", "screen"}:
            analyses = sorted(
                analyses,
                key=lambda item: (
                    item.recommendation == "buy",
                    item.total_return or -999.0,
                    item.sharpe_ratio or -999.0,
                ),
                reverse=True,
            )

        status = "ok" if analyses and not errors else "partial" if analyses else "error"
        return self._response(
            command=command,
            status=status,
            summary=self._summary(command, analyses, errors),
            analyses=analyses,
            errors=errors,
        )

    def _symbols_for(self, payload: TradingAlgoCommandRequest) -> list[str]:
        max_symbols = min(max(int(payload.max_symbols or 1), 1), 25)
        raw_symbols: Iterable[str]
        if payload.command.strip().lower() == "screen" and not payload.symbols:
            raw_symbols = self.DEFAULT_SCREEN_UNIVERSE
        else:
            raw_symbols = payload.symbols or self.DEFAULT_SCREEN_UNIVERSE

        symbols: list[str] = []
        for item in raw_symbols:
            for token in str(item).replace(";", ",").split(","):
                symbol = token.strip().upper()
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
        return (symbols or self.DEFAULT_SCREEN_UNIVERSE)[:max_symbols]

    def _analyze_symbol(self, symbol: str, period: str) -> TradingAlgoSymbolAnalysis:
        history = self.extractor.get_historical_data(symbol=symbol, period=period)
        if history.empty:
            raise ValueError("no historical data returned")

        features = self.extractor.calculate_technical_indicators(history)
        frame = features if not features.empty else history
        latest = frame.iloc[-1]
        close = history["Close"].dropna().astype(float)
        returns = close.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
        risk = self.risk_manager.risk_report(returns)

        latest_price = self._float_or_none(close.iloc[-1] if not close.empty else latest.get("Close"))
        first_price = self._float_or_none(close.iloc[0] if not close.empty else None)
        daily_return = self._float_or_none(returns.iloc[-1]) if not returns.empty else None
        total_return = (
            self._float_or_none((latest_price / first_price) - 1)
            if latest_price is not None and first_price not in (None, 0)
            else None
        )
        sma_20 = self._float_or_none(latest.get("SMA_20", close.tail(20).mean()))
        sma_50 = self._float_or_none(latest.get("SMA_50", close.tail(50).mean()))
        rsi = self._float_or_none(latest.get("RSI"))
        volatility_20d = self._float_or_none(
            latest.get("Volatility_20d", returns.tail(20).std() * math.sqrt(252) if not returns.empty else None)
        )

        trend = self._trend(latest_price, sma_20, sma_50)
        recommendation = self._recommendation(
            trend=trend,
            rsi=rsi,
            total_return=total_return,
            volatility_20d=volatility_20d,
            sharpe_ratio=self._float_or_none(risk.get("sharpe_ratio")),
        )

        return TradingAlgoSymbolAnalysis(
            symbol=symbol,
            period=period,
            rows=int(len(history)),
            as_of=self._index_as_iso(history.index[-1]),
            latest_price=latest_price,
            daily_return=daily_return,
            total_return=total_return,
            volatility_20d=volatility_20d,
            sharpe_ratio=self._float_or_none(risk.get("sharpe_ratio")),
            var_95=self._float_or_none(risk.get("var_95")),
            max_drawdown=self._float_or_none(risk.get("max_drawdown")),
            rsi=rsi,
            sma_20=sma_20,
            sma_50=sma_50,
            trend=trend,
            recommendation=recommendation,
        )

    def _summary(
        self,
        command: str,
        analyses: list[TradingAlgoSymbolAnalysis],
        errors: list[str],
    ) -> str:
        if not analyses:
            return "No trading-algo analysis could be generated."

        buys = sum(1 for item in analyses if item.recommendation == "buy")
        top = analyses[0]
        if command == "screen":
            return f"Screened {len(analyses)} symbols; {buys} buy candidate(s). Top ranked: {top.symbol}."
        if command == "compare":
            return f"Compared {len(analyses)} symbols. Leader: {top.symbol}."
        suffix = f" {len(errors)} symbol(s) failed." if errors else ""
        return f"Analyzed {len(analyses)} symbol(s) with trading_algo.{suffix}"

    @staticmethod
    def _response(
        command: str,
        status: str,
        summary: str,
        analyses: list[TradingAlgoSymbolAnalysis],
        errors: list[str] | None = None,
    ) -> TradingAlgoCommandResponse:
        return TradingAlgoCommandResponse(
            command=command,
            status=status,
            generated_at=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            analyses=analyses,
            errors=errors or [],
        )

    @staticmethod
    def _trend(
        latest_price: float | None,
        sma_20: float | None,
        sma_50: float | None,
    ) -> str:
        if latest_price is None or sma_20 is None or sma_50 is None:
            return "unknown"
        if latest_price >= sma_20 >= sma_50:
            return "bullish"
        if latest_price <= sma_20 <= sma_50:
            return "bearish"
        return "mixed"

    @staticmethod
    def _recommendation(
        trend: str,
        rsi: float | None,
        total_return: float | None,
        volatility_20d: float | None,
        sharpe_ratio: float | None,
    ) -> str:
        score = 0
        score += 2 if trend == "bullish" else -2 if trend == "bearish" else 0
        if rsi is not None:
            score += 1 if 40 <= rsi <= 70 else -1 if rsi >= 78 else 0
        if total_return is not None:
            score += 1 if total_return > 0 else -1
        if volatility_20d is not None:
            score += 1 if volatility_20d < 0.45 else -1
        if sharpe_ratio is not None:
            score += 1 if sharpe_ratio >= 0.75 else -1 if sharpe_ratio < 0 else 0

        if score >= 4:
            return "buy"
        if score >= 1:
            return "watch"
        if score <= -3:
            return "reduce"
        return "hold"

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if math.isnan(result) or math.isinf(result):
            return None
        return round(result, 6)

    @staticmethod
    def _index_as_iso(value: object) -> str | None:
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime().isoformat()
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value) if value is not None else None
