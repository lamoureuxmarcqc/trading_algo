from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from app.schemas.terminal import (
    TradingAlgoBacktestSummary,
    TradingAlgoCommandRequest,
    TradingAlgoCommandResponse,
    TradingAlgoFeatureContribution,
    TradingAlgoModelDiagnostics,
    TradingAlgoPositionSizing,
    TradingAlgoPredictionBand,
    TradingAlgoPredictionInterval,
    TradingAlgoSymbolAnalysis,
)


class TradingAlgoTerminalService:
    """Small API-safe bridge from the institutional terminal to trading_algo."""

    DEFAULT_SCREEN_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "JPM", "XOM", "LLY"]
    ALLOWED_COMMANDS = {"analyze", "compare", "screen", "simulate"}

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
        if command == "simulate":
            return self._run_monte_carlo(payload, symbols)

        analyses: list[TradingAlgoSymbolAnalysis] = []
        errors: list[str] = []

        for symbol in symbols:
            try:
                analyses.append(self.analyze_symbol(symbol, payload.period))
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

    def _run_monte_carlo(
        self,
        payload: TradingAlgoCommandRequest,
        symbols: list[str],
    ) -> TradingAlgoCommandResponse:
        from trading_algo.analytics.simulation import run_monte_carlo

        try:
            returns_map: dict[str, pd.Series] = {}
            for symbol in symbols:
                history = self.extractor.get_historical_data(symbol=symbol, period=payload.period)
                if history.empty:
                    continue
                close = history["Close"].dropna().astype(float)
                daily = close.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
                if not daily.empty:
                    returns_map[symbol] = daily.tail(252)
            if not returns_map:
                raise ValueError("no return history available for simulation")

            frame = pd.DataFrame(returns_map).dropna(how="all").ffill().dropna().tail(252)
            weights = np.array([1.0 / len(frame.columns)] * len(frame.columns))
            np.random.seed(42)
            paths = run_monte_carlo(
                weights=weights,
                returns=frame,
                n_simulations=min(max(int(payload.max_symbols or 8), 100), 1000),
                timeframe=252,
                block_size=20,
                use_stochastic_vol=True,
            )
            final_returns = (paths[-1, :] / 100.0) - 1.0
            var_95 = float(np.percentile(final_returns, 5))
            tail = final_returns[final_returns <= var_95]
            daily_returns = (paths[1:, :] / np.maximum(paths[:-1, :], 1e-9)) - 1.0
            std_daily = float(np.std(daily_returns))
            sharpe = ((float(np.mean(daily_returns)) * 252) / (std_daily * math.sqrt(252))) if std_daily > 0 else 0.0

            analyses = [
                TradingAlgoSymbolAnalysis(
                    symbol=symbol,
                    period=payload.period,
                    rows=int(len(frame)),
                    as_of=self._index_as_iso(frame.index[-1]) if len(frame.index) else None,
                    latest_price=None,
                    daily_return=None,
                    total_return=round(float(np.mean(final_returns)), 4),
                    volatility_20d=round(float(frame[symbol].tail(20).std() * math.sqrt(252)), 4)
                    if symbol in frame.columns
                    else None,
                    sharpe_ratio=round(sharpe, 4),
                    var_95=round(var_95, 4),
                    max_drawdown=None,
                    rsi=None,
                    sma_20=None,
                    sma_50=None,
                    trend="simulated",
                    recommendation="watch",
                )
                for symbol in frame.columns[:1]
            ]
            return self._response(
                command="simulate",
                status="ok",
                summary=(
                    f"Monte Carlo ({paths.shape[1]} paths, 252 days) on {', '.join(frame.columns)}. "
                    f"Expected return {round(float(np.mean(final_returns)) * 100, 2)}%, "
                    f"Sharpe {round(sharpe, 2)}, VaR 95 {round(var_95 * 100, 2)}%."
                ),
                analyses=analyses,
                errors=[],
            )
        except Exception as exc:
            return self._response(
                command="simulate",
                status="error",
                summary="Monte Carlo simulation failed.",
                analyses=[],
                errors=[str(exc)],
            )

    def analyze_symbol(self, symbol: str, period: str = "1y") -> TradingAlgoSymbolAnalysis:
        return self._analyze_symbol(symbol.upper().strip(), period)

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
        score = self._recommendation_score(
            trend=trend,
            rsi=rsi,
            total_return=total_return,
            volatility_20d=volatility_20d,
            sharpe_ratio=self._float_or_none(risk.get("sharpe_ratio")),
        )
        recommendation = self._recommendation(
            score=score,
        )
        buy_probability = self._buy_probability(score, rsi, trend)
        explained_at = datetime.now(timezone.utc).isoformat()
        feature_importance = self._feature_importance(
            latest_price=latest_price,
            sma_20=sma_20,
            sma_50=sma_50,
            rsi=rsi,
            volatility_20d=volatility_20d,
            total_return=total_return,
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
            diagnostics=self._model_diagnostics(
                symbol=symbol,
                frame=frame,
                returns=returns,
                confidence_score=buy_probability,
                trend=trend,
                recommendation=recommendation,
            ),
            explanation_source="heuristic_fallback",
            explained_at=explained_at,
            feature_importance=feature_importance,
            prediction_bands=self._prediction_bands(
                latest_price=latest_price,
                returns=returns,
            ),
            sizing=self._position_sizing(
                latest_price=latest_price,
                buy_probability=buy_probability,
                returns=returns,
                total_return=total_return,
            ),
            backtest=self._backtest_signal(frame=frame, close=close, returns=returns),
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
        score: int,
    ) -> str:
        if score >= 4:
            return "buy"
        if score >= 1:
            return "watch"
        if score <= -3:
            return "reduce"
        return "hold"

    @staticmethod
    def _recommendation_score(
        trend: str,
        rsi: float | None,
        total_return: float | None,
        volatility_20d: float | None,
        sharpe_ratio: float | None,
    ) -> int:
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
        return score

    @staticmethod
    def _buy_probability(score: int, rsi: float | None, trend: str) -> float:
        probability = 0.5 + (score * 0.055)
        if rsi is not None and rsi < 35:
            probability += 0.035
        if trend == "bullish":
            probability += 0.035
        if trend == "bearish":
            probability -= 0.035
        return round(float(np.clip(probability, 0.08, 0.92)), 4)

    def _model_diagnostics(
        self,
        symbol: str,
        frame: pd.DataFrame,
        returns: pd.Series,
        confidence_score: float,
        trend: str = "unknown",
        recommendation: str = "watch",
    ) -> TradingAlgoModelDiagnostics:
        close = frame["Close"].dropna().astype(float) if "Close" in frame else pd.Series(dtype=float)
        signal = self._signal_series(frame)
        next_up = close.pct_change().shift(-1) > 0
        predictions = signal > 0

        directional_accuracy_30d = self._directional_accuracy(predictions, next_up, 30)
        directional_accuracy_90d = self._directional_accuracy(predictions, next_up, 90)
        predicted_next = close.shift(1) * (1 + returns.rolling(20, min_periods=5).mean().shift(1))
        errors = (close - predicted_next).dropna().tail(90)
        strategy_returns = (signal.shift(1).reindex(returns.index).fillna(0) * returns).dropna()
        latest_timestamp = frame.index[-1] if len(frame.index) else None
        latest_age_hours = self._timestamp_age_hours(latest_timestamp)
        volatility_ratio = self._rolling_ratio(returns.tail(20).std(), returns.tail(90).std())

        return TradingAlgoModelDiagnostics(
            model_version=f"technical-ensemble-{symbol.lower()}-v1",
            feature_set_version="ohlcv-technical-v1",
            trained_at=self._index_as_iso(frame.index[-60]) if len(frame.index) >= 60 else self._index_as_iso(frame.index[0]),
            inference_at=datetime.now(timezone.utc).isoformat(),
            confidence_score=confidence_score,
            directional_accuracy_30d=directional_accuracy_30d,
            directional_accuracy_90d=directional_accuracy_90d,
            hit_rate=directional_accuracy_90d,
            mae=self._float_or_none(errors.abs().mean()) if not errors.empty else None,
            rmse=self._float_or_none(math.sqrt(float((errors**2).mean()))) if not errors.empty else None,
            signal_sharpe_30d=self._signal_sharpe(strategy_returns.tail(30)),
            signal_sharpe_90d=self._signal_sharpe(strategy_returns.tail(90)),
            model_stale=latest_age_hours is None or latest_age_hours > 24,
            data_drift_detected=bool(volatility_ratio is not None and volatility_ratio > 1.75),
            concept_drift_detected=bool(
                directional_accuracy_30d is not None
                and directional_accuracy_90d is not None
                and directional_accuracy_30d + 0.12 < directional_accuracy_90d
            ),
            market_data_stale=latest_age_hours is None or latest_age_hours > 24,
            regime_mismatch=bool(trend == "bearish" and recommendation == "buy"),
        )

    def _feature_importance(
        self,
        latest_price: float | None,
        sma_20: float | None,
        sma_50: float | None,
        rsi: float | None,
        volatility_20d: float | None,
        total_return: float | None,
    ) -> list[TradingAlgoFeatureContribution]:
        items: list[TradingAlgoFeatureContribution] = []
        if latest_price is not None and sma_20 not in (None, 0):
            contribution = float(np.clip((latest_price / sma_20 - 1) * 2.5, -0.18, 0.18))
            items.append(
                TradingAlgoFeatureContribution(
                    feature="Momentum vs SMA 20",
                    contribution=round(contribution, 4),
                    direction="positive" if contribution >= 0 else "negative",
                    explanation="Price trading above the 20-day average supports the signal; below it weakens momentum.",
                )
            )
        if sma_20 is not None and sma_50 not in (None, 0):
            contribution = float(np.clip((sma_20 / sma_50 - 1) * 2.0, -0.14, 0.14))
            items.append(
                TradingAlgoFeatureContribution(
                    feature="Trend slope 20/50",
                    contribution=round(contribution, 4),
                    direction="positive" if contribution >= 0 else "negative",
                    explanation="The 20-day average relative to the 50-day average estimates medium-term trend quality.",
                )
            )
        if rsi is not None:
            contribution = float(np.clip((65 - abs(rsi - 55)) / 650, -0.08, 0.10))
            if rsi > 76:
                contribution = -0.07
            items.append(
                TradingAlgoFeatureContribution(
                    feature="RSI balance",
                    contribution=round(contribution, 4),
                    direction="positive" if contribution >= 0 else "negative",
                    explanation="Balanced RSI improves confidence while overbought readings reduce forward edge.",
                )
            )
        if volatility_20d is not None:
            contribution = float(np.clip(0.06 - volatility_20d * 0.18, -0.10, 0.06))
            items.append(
                TradingAlgoFeatureContribution(
                    feature="Volatility 20D",
                    contribution=round(contribution, 4),
                    direction="positive" if contribution >= 0 else "negative",
                    explanation="Lower realized volatility supports position sizing; high volatility compresses edge.",
                )
            )
        if total_return is not None:
            contribution = float(np.clip(total_return * 0.18, -0.08, 0.10))
            items.append(
                TradingAlgoFeatureContribution(
                    feature="Lookback return",
                    contribution=round(contribution, 4),
                    direction="positive" if contribution >= 0 else "negative",
                    explanation="Positive lookback return adds trend confirmation; negative return is a drag.",
                )
            )
        return sorted(items, key=lambda item: abs(item.contribution), reverse=True)[:5]

    def _prediction_bands(
        self,
        latest_price: float | None,
        returns: pd.Series,
    ) -> list[TradingAlgoPredictionBand]:
        if latest_price is None or returns.empty:
            return []

        daily_mean = float(returns.tail(90).mean() if len(returns) >= 20 else returns.mean())
        daily_vol = float(returns.tail(90).std() if len(returns) >= 20 else returns.std())
        if math.isnan(daily_vol) or daily_vol <= 0:
            daily_vol = 0.01

        bands: list[TradingAlgoPredictionBand] = []
        annual_vol = daily_vol * math.sqrt(252)
        stop_return = -max(0.04, min(0.18, annual_vol / math.sqrt(252) * 8))
        target_return = max(0.08, min(0.35, abs(daily_mean * 252) * 0.35 + annual_vol * 0.25))
        for horizon in [1, 5, 30]:
            expected_return = daily_mean * horizon
            median = latest_price * (1 + expected_return)
            horizon_vol = daily_vol * math.sqrt(horizon)
            intervals = []
            for confidence, z_score in [(0.50, 0.674), (0.80, 1.282), (0.95, 1.96)]:
                band_width = z_score * horizon_vol
                intervals.append(
                    TradingAlgoPredictionInterval(
                        confidence=confidence,
                        low=round(latest_price * (1 + expected_return - band_width), 4),
                        high=round(latest_price * (1 + expected_return + band_width), 4),
                    )
                )
            target_price = latest_price * (1 + target_return)
            stop_price = latest_price * (1 + stop_return)
            bands.append(
                TradingAlgoPredictionBand(
                    horizon_days=horizon,
                    median_target=round(median, 4),
                    expected_return=round(expected_return, 6),
                    target_price=round(target_price, 4),
                    stop_price=round(stop_price, 4),
                    probability_of_target=self._barrier_hit_probability(
                        expected_return, horizon_vol, target_return, upper=True
                    ),
                    probability_of_stop=self._barrier_hit_probability(
                        expected_return, horizon_vol, stop_return, upper=False
                    ),
                    intervals=intervals,
                )
            )
        return bands

    def _position_sizing(
        self,
        latest_price: float | None,
        buy_probability: float,
        returns: pd.Series,
        total_return: float | None,
    ) -> TradingAlgoPositionSizing | None:
        if latest_price is None:
            return None
        realized_vol = float(returns.tail(20).std() * math.sqrt(252)) if not returns.empty else 0.25
        realized_vol = max(realized_vol, 0.05)
        model_edge = float(np.clip((buy_probability - 0.5) + ((total_return or 0.0) * 0.25), -0.25, 0.35))
        variance = max(realized_vol**2, 0.01)
        kelly_fraction = float(np.clip(model_edge / variance, -0.5, 0.5))
        capped_kelly = float(np.clip(kelly_fraction, 0.0, 0.15))
        volatility_adjusted = float(np.clip(0.18 / realized_vol * capped_kelly, 0.0, 0.12))
        recommended_weight = min(capped_kelly, volatility_adjusted)
        recommended_notional = 1_000_000 * recommended_weight
        return TradingAlgoPositionSizing(
            buy_probability=round(buy_probability, 4),
            model_edge=round(model_edge, 4),
            kelly_fraction=round(kelly_fraction, 4),
            capped_kelly_fraction=round(capped_kelly, 4),
            volatility_adjusted_weight=round(volatility_adjusted, 4),
            recommended_weight=round(recommended_weight, 4),
            recommended_notional=round(recommended_notional, 2),
            recommended_quantity=round(recommended_notional / latest_price, 4) if latest_price else None,
            risk_budget_consumed=round(min(recommended_weight / 0.12, 1.0), 4),
            stop_loss=round(max(0.04, min(0.18, realized_vol / math.sqrt(252) * 8)), 4),
            take_profit=round(max(0.08, min(0.35, abs(model_edge) * 1.8 + realized_vol * 0.25)), 4),
        )

    def _backtest_signal(
        self,
        frame: pd.DataFrame,
        close: pd.Series,
        returns: pd.Series,
    ) -> TradingAlgoBacktestSummary | None:
        if returns.empty or close.empty:
            return None
        lookback = min(90, len(returns))
        signal = self._signal_series(frame).reindex(returns.index).fillna(0)
        strategy_returns = (signal.shift(1).fillna(0) * returns).tail(lookback)
        if strategy_returns.empty:
            return None
        equity = (1 + strategy_returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        exposed = signal.tail(lookback) > 0
        trade_count = int((exposed.astype(int).diff().fillna(0) > 0).sum())
        wins = int((strategy_returns[strategy_returns != 0] > 0).sum())
        losses = int((strategy_returns[strategy_returns != 0] < 0).sum())
        active_trades = strategy_returns[strategy_returns != 0]
        pnl = float(equity.iloc[-1] - 1)
        benchmark_return = float((1 + returns.tail(lookback)).prod() - 1)
        downside = strategy_returns[strategy_returns < 0]
        sortino = None
        if not downside.empty and float(downside.std()) > 0:
            sortino = float(strategy_returns.mean() / downside.std() * math.sqrt(252))
        return TradingAlgoBacktestSummary(
            lookback_days=lookback,
            pnl=round(pnl, 6),
            annualized_return=round(float((1 + pnl) ** (252 / max(lookback, 1)) - 1), 6),
            max_drawdown=round(float(drawdown.min()), 6),
            win_loss_ratio=round(float(wins / max(losses, 1)), 4),
            trade_count=trade_count,
            exposure_time=round(float(exposed.mean()), 4),
            average_trade=round(float(active_trades.mean()), 6) if not active_trades.empty else 0.0,
            fees=0.0,
            slippage=0.0,
            benchmark_return=round(benchmark_return, 6),
            excess_return=round(pnl - benchmark_return, 6),
            sharpe_ratio=self._signal_sharpe(strategy_returns),
            sortino_ratio=round(sortino, 4) if sortino is not None else None,
        )

    def backtest_with_policy(
        self,
        symbol: str,
        *,
        period: str,
        lookback_days: int,
        buy_probability_threshold: float,
        stop_loss: float,
        take_profit: float,
        slippage_bps: float,
        max_position_weight: float,
    ) -> TradingAlgoBacktestSummary:
        analysis = self.analyze_symbol(symbol, period)
        history = self.extractor.get_historical_data(symbol=symbol, period=period)
        if history.empty:
            raise ValueError("no historical data returned")
        features = self.extractor.calculate_technical_indicators(history)
        frame = features if not features.empty else history
        close = history["Close"].dropna().astype(float)
        returns = close.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
        lookback = min(max(int(lookback_days), 5), len(returns))
        raw_signal = self._signal_series(frame).reindex(returns.index).fillna(0)
        buy_gate = 1.0 if (analysis.sizing and analysis.sizing.buy_probability >= buy_probability_threshold) else 0.0
        signal = (raw_signal * buy_gate).clip(upper=1.0)
        slippage = slippage_bps / 10_000.0
        entries = (signal.shift(1).fillna(0) > 0) & (signal.shift(2).fillna(0) == 0)
        strategy_returns = (signal.shift(1).fillna(0) * returns * max_position_weight).tail(lookback)
        costs = entries.reindex(strategy_returns.index).fillna(False).astype(float) * (slippage + 0.0005)
        strategy_returns = strategy_returns - costs
        if stop_loss > 0 or take_profit > 0:
            clipped = strategy_returns.clip(lower=-abs(stop_loss) * max_position_weight, upper=abs(take_profit) * max_position_weight)
            strategy_returns = clipped
        if strategy_returns.empty:
            raise ValueError("insufficient history for backtest")
        equity = (1 + strategy_returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        exposed = signal.tail(lookback) > 0
        trade_count = int((exposed.astype(int).diff().fillna(0) > 0).sum())
        wins = int((strategy_returns[strategy_returns != 0] > 0).sum())
        losses = int((strategy_returns[strategy_returns != 0] < 0).sum())
        active_trades = strategy_returns[strategy_returns != 0]
        pnl = float(equity.iloc[-1] - 1)
        benchmark_return = float((1 + returns.tail(lookback)).prod() - 1)
        downside = strategy_returns[strategy_returns < 0]
        sortino = None
        if not downside.empty and float(downside.std()) > 0:
            sortino = float(strategy_returns.mean() / downside.std() * math.sqrt(252))
        return TradingAlgoBacktestSummary(
            lookback_days=lookback,
            pnl=round(pnl, 6),
            annualized_return=round(float((1 + pnl) ** (252 / max(lookback, 1)) - 1), 6),
            max_drawdown=round(float(drawdown.min()), 6),
            win_loss_ratio=round(float(wins / max(losses, 1)), 4),
            trade_count=trade_count,
            exposure_time=round(float(exposed.mean()), 4),
            average_trade=round(float(active_trades.mean()), 6) if not active_trades.empty else 0.0,
            fees=round(float(entries.tail(lookback).sum() * 0.0005), 6),
            slippage=round(float(entries.tail(lookback).sum() * slippage), 6),
            benchmark_return=round(benchmark_return, 6),
            excess_return=round(pnl - benchmark_return, 6),
            sharpe_ratio=self._signal_sharpe(strategy_returns),
            sortino_ratio=round(sortino, 4) if sortino is not None else None,
        )

    @staticmethod
    def parse_lookback_days(lookback: str, default: int = 180) -> int:
        token = str(lookback or "").strip().lower()
        if token.endswith("d") and token[:-1].isdigit():
            return max(int(token[:-1]), 5)
        if token.endswith("mo") and token[:-2].isdigit():
            return max(int(token[:-2]) * 21, 5)
        if token.endswith("y") and token[:-1].isdigit():
            return max(int(token[:-1]) * 252, 5)
        if token.isdigit():
            return max(int(token), 5)
        return default

    @staticmethod
    def warning_flags(diagnostics: TradingAlgoModelDiagnostics | None) -> list[str]:
        if diagnostics is None:
            return ["model_unavailable"]
        flags = []
        if diagnostics.model_stale:
            flags.append("model_stale")
        if diagnostics.market_data_stale:
            flags.append("market_data_stale")
        if diagnostics.data_drift_detected:
            flags.append("data_drift_detected")
        if diagnostics.concept_drift_detected:
            flags.append("concept_drift_detected")
        if diagnostics.regime_mismatch:
            flags.append("regime_mismatch")
        return flags

    @staticmethod
    def _barrier_hit_probability(expected_return: float, horizon_vol: float, barrier: float, *, upper: bool) -> float:
        if horizon_vol <= 0:
            hit = expected_return >= barrier if upper else expected_return <= barrier
            return 1.0 if hit else 0.0
        z_score = (barrier - expected_return) / horizon_vol
        cdf = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        probability = 1.0 - cdf if upper else cdf
        return round(float(np.clip(probability, 0.0, 1.0)), 4)

    @staticmethod
    def _signal_series(frame: pd.DataFrame) -> pd.Series:
        close = frame["Close"].astype(float)
        sma_20 = frame["SMA_20"].astype(float) if "SMA_20" in frame else close.rolling(20, min_periods=1).mean()
        sma_50 = frame["SMA_50"].astype(float) if "SMA_50" in frame else close.rolling(50, min_periods=1).mean()
        rsi = frame["RSI"].astype(float) if "RSI" in frame else pd.Series(55.0, index=frame.index)
        return ((close > sma_20) & (sma_20 >= sma_50) & (rsi < 76)).astype(float)

    @staticmethod
    def _directional_accuracy(predictions: pd.Series, actuals: pd.Series, lookback: int) -> float | None:
        aligned = pd.concat([predictions, actuals], axis=1).dropna().tail(lookback)
        if aligned.empty:
            return None
        return round(float((aligned.iloc[:, 0] == aligned.iloc[:, 1]).mean()), 4)

    @staticmethod
    def _signal_sharpe(strategy_returns: pd.Series) -> float | None:
        clean = strategy_returns.replace([math.inf, -math.inf], pd.NA).dropna()
        if clean.empty or float(clean.std()) == 0:
            return None
        return round(float(clean.mean() / clean.std() * math.sqrt(252)), 4)

    @staticmethod
    def _rolling_ratio(numerator: object, denominator: object) -> float | None:
        try:
            den = float(denominator)
            num = float(numerator)
        except (TypeError, ValueError):
            return None
        if den <= 0 or math.isnan(den) or math.isnan(num):
            return None
        return num / den

    @staticmethod
    def _timestamp_age_hours(value: object) -> float | None:
        if value is None:
            return None
        if hasattr(value, "to_pydatetime"):
            timestamp = value.to_pydatetime()
        elif isinstance(value, datetime):
            timestamp = value
        else:
            try:
                timestamp = pd.Timestamp(value).to_pydatetime()
            except Exception:
                return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds() / 3600)

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
