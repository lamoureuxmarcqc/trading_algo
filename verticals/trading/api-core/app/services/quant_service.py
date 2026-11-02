from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from trading_algo.data.data_extraction import MacroDataExtractor, MarketHealthAnalyzer, StockDataExtractor
from trading_algo.risk.risk_manager import RiskManager
from trading_algo.strategy.market_regime_engine import MarketRegimeEngine


@dataclass
class CachedFrame:
    data: pd.DataFrame
    timestamp: datetime


@dataclass
class CachedSnapshot:
    data: dict[str, dict[str, float | None]]
    timestamp: datetime


class QuantInsightsService:
    def __init__(self) -> None:
        self.extractor = StockDataExtractor()
        self.macro_extractor = MacroDataExtractor()
        self.health_analyzer = MarketHealthAnalyzer(self.macro_extractor)
        self.regime_engine = MarketRegimeEngine()
        self.risk_manager = RiskManager()
        self._history_cache: dict[tuple[str, str], CachedFrame] = {}
        self._snapshot_cache: dict[tuple[tuple[str, str], ...], CachedSnapshot] = {}
        self._cache_ttl = timedelta(minutes=20)
        self._snapshot_ttl = timedelta(seconds=60)
        self._research_universe = [
            {"symbol": "AAPL", "sector": "Technology"},
            {"symbol": "MSFT", "sector": "Technology"},
            {"symbol": "NVDA", "sector": "Semiconductors"},
            {"symbol": "AMZN", "sector": "Consumer"},
            {"symbol": "META", "sector": "Communication"},
            {"symbol": "JPM", "sector": "Financials"},
            {"symbol": "XOM", "sector": "Energy"},
            {"symbol": "LLY", "sector": "Healthcare"},
        ]

    def get_live_market_snapshot(self, symbols: list[str]) -> dict[str, dict[str, float | None]]:
        return self.get_live_market_snapshot_for_requests({symbol: symbol for symbol in symbols})

    def get_live_market_snapshot_for_requests(
        self,
        symbol_requests: dict[str, str | None],
    ) -> dict[str, dict[str, float | None]]:
        if not symbol_requests:
            return {}

        cache_key = tuple(sorted((symbol, request or "") for symbol, request in symbol_requests.items()))
        cached = self._snapshot_cache.get(cache_key)
        if cached and datetime.now() - cached.timestamp < self._snapshot_ttl:
            return dict(cached.data)

        request_symbols = sorted({request for request in symbol_requests.values() if request})
        prices = self.extractor.get_bulk_prices(request_symbols) if request_symbols else {}
        snapshot: dict[str, dict[str, float | None]] = {}

        for symbol, request_symbol in symbol_requests.items():
            if not request_symbol:
                snapshot[symbol] = {"latest_price": None, "previous_close": None}
                continue

            latest_price = prices.get(request_symbol)
            history = self._history(
                symbol,
                period="1mo",
                market_data_symbol=request_symbol,
                market_data_enabled=True,
            )
            prev_close = None
            if not history.empty and "Close" in history.columns:
                closes = history["Close"].dropna()
                if len(closes) >= 2:
                    prev_close = float(closes.iloc[-2])
                elif len(closes) == 1:
                    prev_close = float(closes.iloc[-1])
                if latest_price in (None, 0.0):
                    latest_price = float(closes.iloc[-1])

            if latest_price in (None, 0.0):
                latest_price = None

            snapshot[symbol] = {
                "latest_price": float(latest_price) if latest_price is not None else None,
                "previous_close": float(prev_close) if prev_close is not None else None,
            }

        self._snapshot_cache[cache_key] = CachedSnapshot(data=dict(snapshot), timestamp=datetime.now())
        return snapshot

    def compute_portfolio_risk(
        self,
        positions: list[Any],
        gross_exposure: float,
        net_exposure: float,
    ) -> dict[str, float]:
        if not positions:
            return {
                "var_95": -0.023,
                "cvar_95": -0.034,
                "beta": 1.08,
                "drawdown": -0.081,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "concentration_risk": 0.0,
                "correlation_risk": 0.62,
            }
        returns_map, weights = self._portfolio_returns(positions)
        if not returns_map or not weights:
            return {
                "var_95": -0.023,
                "cvar_95": -0.034,
                "beta": 1.08,
                "drawdown": -0.081,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "concentration_risk": self._max_weight(positions),
                "correlation_risk": 0.62,
            }

        returns_df = pd.DataFrame(returns_map).dropna(how="all").ffill().dropna()
        if returns_df.empty:
            return {
                "var_95": -0.023,
                "cvar_95": -0.034,
                "beta": 1.08,
                "drawdown": -0.081,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "concentration_risk": self._max_weight(positions),
                "correlation_risk": 0.62,
            }

        ordered_symbols = list(returns_df.columns)
        aligned_weights = np.array([weights[symbol] for symbol in ordered_symbols], dtype=float)
        aligned_weights = aligned_weights / aligned_weights.sum()
        portfolio_returns = returns_df.dot(aligned_weights)

        benchmark_returns = self._returns_series(settings.default_benchmark, period="1y")
        benchmark_aligned = self._align_series(portfolio_returns, benchmark_returns)
        risk_report = self.risk_manager.risk_report(portfolio_returns, benchmark_aligned)
        cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        correlation_risk = self._correlation_risk(returns_df)

        return {
            "var_95": float(risk_report.get("var_95", -0.023)),
            "cvar_95": cvar_95,
            "beta": float(risk_report.get("beta", 1.0)),
            "drawdown": float(risk_report.get("max_drawdown", -0.081)),
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "concentration_risk": self._max_weight(positions),
            "correlation_risk": correlation_risk,
        }

    def compute_position_risks(self, positions: list[Any]) -> list[dict[str, float | str]]:
        benchmark_returns = self._returns_series(settings.default_benchmark, period="1y")
        total_value = sum(max(float(position.market_value), 0.0) for position in positions)
        results: list[dict[str, float | str]] = []

        for position in positions:
            request_symbol = self._position_market_data_symbol(position)
            returns = self._returns_series(
                position.symbol,
                period="1y",
                market_data_symbol=request_symbol,
                market_data_enabled=self._position_market_data_enabled(position),
            )
            beta = self.risk_manager.calculate_beta(returns, self._align_series(returns, benchmark_returns))
            var_95 = self.risk_manager.calculate_value_at_risk(returns, 0.95)
            cvar_95 = self._calculate_cvar(returns, 0.95)
            liquidity_score = self._liquidity_score(
                position.symbol,
                market_data_symbol=request_symbol,
                market_data_enabled=self._position_market_data_enabled(position),
            )
            weight = (float(position.market_value) / total_value) if total_value else 0.0

            results.append(
                {
                    "symbol": position.symbol,
                    "beta": round(float(beta), 2),
                    "var_95": round(float(var_95), 4),
                    "cvar_95": round(float(cvar_95), 4),
                    "liquidity_score": round(liquidity_score, 2),
                    "concentration_weight": round(weight, 4),
                }
            )
        return results

    def compute_correlation_matrix(self, positions: list[Any]) -> dict[str, Any]:
        symbols = [position.symbol for position in positions]
        if not symbols:
            return {
                "symbols": [],
                "matrix": [],
                "as_of": None,
                "methodology": "Daily close-to-close return correlation over the last 1 year.",
            }

        returns_map, _ = self._portfolio_returns(positions)
        if not returns_map:
            return self._fallback_correlation_matrix(symbols)

        returns_df = pd.DataFrame(returns_map).dropna(how="all").ffill().dropna()
        if returns_df.empty:
            return self._fallback_correlation_matrix(symbols)

        correlation = returns_df.corr().fillna(0.0).clip(-1.0, 1.0)
        ordered_symbols = list(correlation.columns)
        matrix = [
            [round(float(correlation.loc[row_symbol, col_symbol]), 4) for col_symbol in ordered_symbols]
            for row_symbol in ordered_symbols
        ]

        as_of = None
        if len(returns_df.index) > 0:
            try:
                as_of = pd.Timestamp(returns_df.index.max()).to_pydatetime().isoformat()
            except Exception:
                as_of = None

        return {
            "symbols": ordered_symbols,
            "matrix": matrix,
            "as_of": as_of,
            "methodology": "Daily close-to-close return correlation over the last 1 year.",
        }

    def compute_signal(
        self,
        symbol: str,
        current_price: float | None = None,
        market_data_symbol: str | None = None,
        market_data_enabled: bool = True,
    ) -> dict[str, float | str]:
        df = self._history(
            symbol,
            period="1y",
            market_data_symbol=market_data_symbol,
            market_data_enabled=market_data_enabled,
        )
        regime = self.compute_regime()
        if df.empty or "Close" not in df.columns:
            return {
                "symbol": symbol,
                "buy_probability": 0.56,
                "sell_probability": 0.44,
                "volatility_forecast": 0.19,
                "confidence_score": regime["confidence"],
                "market_regime": regime["regime"],
            }

        indicators = self.extractor.calculate_technical_indicators(df)
        closes = df["Close"].dropna()
        returns = closes.pct_change().dropna()
        latest_close = float(closes.iloc[-1]) if not closes.empty else float(current_price or 0.0)
        sma_20 = self._latest_value(indicators, "SMA_20", latest_close)
        sma_50 = self._latest_value(indicators, "SMA_50", latest_close)
        rsi = self._latest_value(indicators, "RSI", 50.0)
        macd_hist = self._latest_value(indicators, "MACD_Histogram", 0.0)
        momentum_20 = self._latest_value(indicators, "Momentum_20", 0.0)
        volatility_20 = self._latest_value(indicators, "Volatility_20d", 0.2)

        trend_20 = np.clip((latest_close / max(sma_20, 1e-6)) - 1, -0.15, 0.15)
        trend_50 = np.clip((latest_close / max(sma_50, 1e-6)) - 1, -0.20, 0.20)
        score = 0.0
        score += float(np.tanh(trend_20 * 8)) * 0.10
        score += float(np.tanh(trend_50 * 6)) * 0.10
        score += 0.05 if macd_hist > 0 else -0.05
        score += float(np.clip(momentum_20 * 2.5, -0.08, 0.08))

        if rsi < 35:
            score += 0.06
        elif rsi > 70:
            score -= 0.08
        elif 48 <= rsi <= 62:
            score += 0.03

        if regime["regime"] == "risk_on":
            score += 0.04
        elif regime["regime"] in {"risk_off", "defensive"}:
            score -= 0.05

        if not np.isnan(volatility_20):
            if volatility_20 > 0.45:
                score -= 0.04
            elif volatility_20 < 0.18:
                score += 0.02

        buy_probability = float(np.clip(0.5 + score, 0.32, 0.72))
        sell_probability = float(np.clip(1 - buy_probability, 0.28, 0.68))
        confidence_score = float(np.clip(0.48 + abs(score) * 1.2, 0.48, 0.82))
        volatility_forecast = float(
            np.clip(
                volatility_20 if not np.isnan(volatility_20) else returns.std() * np.sqrt(252),
                0.10,
                0.65,
            )
        )

        return {
            "symbol": symbol,
            "buy_probability": round(buy_probability, 4),
            "sell_probability": round(sell_probability, 4),
            "volatility_forecast": round(volatility_forecast, 4),
            "confidence_score": round(confidence_score, 4),
            "market_regime": str(regime["regime"]),
        }

    def compute_forecast(
        self,
        symbol: str,
        current_price: float | None = None,
        market_data_symbol: str | None = None,
        market_data_enabled: bool = True,
    ) -> dict[str, float | str]:
        df = self._history(
            symbol,
            period="1y",
            market_data_symbol=market_data_symbol,
            market_data_enabled=market_data_enabled,
        )
        regime = self.compute_regime()
        if df.empty or "Close" not in df.columns:
            base_price = float(current_price or 250.0)
            return {
                "symbol": symbol,
                "price_target": round(base_price * 1.03, 2),
                "expected_return": 0.03,
                "confidence_interval_low": round(base_price * 0.95, 2),
                "confidence_interval_high": round(base_price * 1.08, 2),
            }

        closes = df["Close"].dropna()
        returns = closes.pct_change().dropna()
        hist_price = float(closes.iloc[-1])
        base_price = float(current_price) if current_price is not None and current_price > 0 else hist_price
        ret_20 = float(closes.pct_change(20).iloc[-1]) if len(closes) > 20 else 0.0
        ret_60 = float(closes.pct_change(60).iloc[-1]) if len(closes) > 60 else ret_20
        ret_120 = float(closes.pct_change(120).iloc[-1]) if len(closes) > 120 else ret_60
        annualized_vol = float(np.clip(returns.std() * np.sqrt(252), 0.10, 0.75)) if not returns.empty else 0.20

        trend_return = ret_20 * 0.5 + ret_60 * 0.3 + ret_120 * 0.2
        dampener = max(0.55, 1 - annualized_vol * 0.55)
        regime_bias = 0.01 if regime["regime"] == "risk_on" else -0.01 if regime["regime"] in {"risk_off", "defensive"} else 0.0
        expected_return = float(np.clip(trend_return * dampener + regime_bias, -0.08, 0.10))
        band = float(np.clip((annualized_vol / np.sqrt(12)) * 0.55, 0.04, 0.12))

        return {
            "symbol": symbol,
            "price_target": round(base_price * (1 + expected_return), 2),
            "expected_return": round(expected_return, 4),
            "confidence_interval_low": round(base_price * (1 + expected_return - band), 2),
            "confidence_interval_high": round(base_price * (1 + expected_return + band), 2),
        }

    def compute_regime(self) -> dict[str, float | str]:
        sp500 = self._history("^GSPC", period="6mo")
        nasdaq = self._history("^IXIC", period="6mo")
        vix = self._history("^VIX", period="6mo")
        treasury_10y = self._history("^TNX", period="6mo")
        indices = {
            "^GSPC": {"closes": sp500["Close"].dropna().tolist()} if not sp500.empty else {"closes": []},
            "^IXIC": {"closes": nasdaq["Close"].dropna().tolist()} if not nasdaq.empty else {"closes": []},
        }

        rates = 4.0
        if not treasury_10y.empty and "Close" in treasury_10y.columns:
            raw_rate = float(treasury_10y["Close"].dropna().iloc[-1])
            rates = raw_rate / 10 if raw_rate > 20 else raw_rate
        macro = {"inflation": 2.8, "rates": rates}

        try:
            regime_label, regime_score = self.regime_engine.compute_regime(macro, indices)
        except Exception:
            regime_label, regime_score = "NEUTRAL", 40

        health_score = self._market_health_score(sp500, nasdaq, vix, rates)
        blended_score = float(np.clip((float(regime_score) * 0.45) + (health_score * 0.55), 0, 100))

        if regime_label in {"RISK ON", "BULLISH"} or blended_score >= 65:
            regime = "risk_on"
            recommendation = "Scale into winners"
        elif regime_label == "DEFENSIVE" or blended_score < 35:
            regime = "defensive"
            recommendation = "Favor quality and hedges"
        elif regime_label == "RISK OFF":
            regime = "risk_off"
            recommendation = "Reduce beta and tighten risk"
        else:
            regime = "neutral"
            recommendation = "Stay selective and balanced"

        confidence = float(np.clip(0.52 + abs(blended_score - 50) / 180, 0.52, 0.80))
        return {
            "regime": regime,
            "confidence": round(confidence, 4),
            "recommendation": recommendation,
            "score": round(blended_score, 2),
        }

    def build_research_screener(self) -> list[dict[str, float | str]]:
        symbols = [item["symbol"] for item in self._research_universe]
        sectors = {item["symbol"]: item["sector"] for item in self._research_universe}
        snapshot = self.get_live_market_snapshot(symbols)
        ideas: list[dict[str, float | str]] = []

        for symbol in symbols:
            current_price = snapshot.get(symbol, {}).get("latest_price")
            signal = self.compute_signal(symbol, current_price=current_price)
            forecast = self.compute_forecast(symbol, current_price=current_price)
            factor_score = self._factor_score(symbol)
            ideas.append(
                {
                    "symbol": symbol,
                    "sector": sectors[symbol],
                    "price": round(float(current_price or forecast["price_target"]), 2),
                    "buy_probability": float(signal["buy_probability"]),
                    "expected_return": float(forecast["expected_return"]),
                    "confidence_score": float(signal["confidence_score"]),
                    "factor_score": round(factor_score, 4),
                    "market_regime": str(signal["market_regime"]),
                }
            )

        return sorted(
            ideas,
            key=lambda item: (
                float(item["buy_probability"]) * 0.45
                + float(item["expected_return"]) * 2.2
                + float(item["factor_score"]) * 0.35
            ),
            reverse=True,
        )

    def build_factor_ranking(self) -> list[dict[str, float | str]]:
        rankings: list[dict[str, float | str]] = []
        for item in self._research_universe:
            symbol = item["symbol"]
            sector = item["sector"]
            metrics = self._factor_metrics(symbol)
            overall = (
                metrics["momentum_score"] * 0.45
                + metrics["quality_score"] * 0.35
                + metrics["volatility_score"] * 0.20
            )
            rankings.append(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "momentum_score": round(metrics["momentum_score"], 4),
                    "quality_score": round(metrics["quality_score"], 4),
                    "volatility_score": round(metrics["volatility_score"], 4),
                    "overall_score": round(overall, 4),
                }
            )
        return sorted(rankings, key=lambda item: float(item["overall_score"]), reverse=True)

    def build_sector_rotation(self) -> list[dict[str, float | str]]:
        screener = self.build_research_screener()
        grouped: dict[str, list[dict[str, float | str]]] = {}
        for idea in screener:
            grouped.setdefault(str(idea["sector"]), []).append(idea)

        results: list[dict[str, float | str]] = []
        for sector, items in grouped.items():
            avg_buy = float(np.mean([float(item["buy_probability"]) for item in items]))
            avg_return = float(np.mean([float(item["expected_return"]) for item in items]))
            avg_factor = float(np.mean([float(item["factor_score"]) for item in items]))
            score = avg_buy * 0.45 + avg_return * 2.2 + avg_factor * 0.35
            if score >= 0.62:
                stance = "overweight"
            elif score <= 0.48:
                stance = "underweight"
            else:
                stance = "market_weight"
            results.append(
                {
                    "sector": sector,
                    "average_buy_probability": round(avg_buy, 4),
                    "average_expected_return": round(avg_return, 4),
                    "average_factor_score": round(avg_factor, 4),
                    "stance": stance,
                }
            )
        return sorted(results, key=lambda item: float(item["average_factor_score"]), reverse=True)

    def _history(
        self,
        symbol: str,
        period: str = "1y",
        market_data_symbol: str | None = None,
        market_data_enabled: bool = True,
    ) -> pd.DataFrame:
        request_symbol = self._resolve_market_data_symbol(symbol, market_data_symbol, market_data_enabled)
        if not request_symbol:
            return pd.DataFrame()

        key = (request_symbol, period)
        cached = self._history_cache.get(key)
        if cached and datetime.now() - cached.timestamp < self._cache_ttl:
            return cached.data.copy()

        df = self.extractor.get_historical_data(request_symbol, period=period)
        if df is None or df.empty:
            return pd.DataFrame()

        self._history_cache[key] = CachedFrame(data=df.copy(), timestamp=datetime.now())
        return df.copy()

    def _returns_series(
        self,
        symbol: str,
        period: str = "1y",
        market_data_symbol: str | None = None,
        market_data_enabled: bool = True,
    ) -> pd.Series:
        df = self._history(
            symbol,
            period=period,
            market_data_symbol=market_data_symbol,
            market_data_enabled=market_data_enabled,
        )
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        returns = df["Close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        returns.name = symbol
        return returns

    def _portfolio_returns(self, positions: list[Any]) -> tuple[dict[str, pd.Series], dict[str, float]]:
        total_value = sum(max(float(position.market_value), 0.0) for position in positions)
        if total_value <= 0:
            return {}, {}

        returns_map: dict[str, pd.Series] = {}
        weights: dict[str, float] = {}
        for position in positions:
            request_symbol = self._position_market_data_symbol(position)
            returns = self._returns_series(
                position.symbol,
                period="1y",
                market_data_symbol=request_symbol,
                market_data_enabled=self._position_market_data_enabled(position),
            )
            if returns.empty:
                continue
            returns_map[position.symbol] = returns
            weights[position.symbol] = float(position.market_value) / total_value
        return returns_map, weights

    def _factor_metrics(
        self,
        symbol: str,
        market_data_symbol: str | None = None,
        market_data_enabled: bool = True,
    ) -> dict[str, float]:
        df = self._history(
            symbol,
            period="1y",
            market_data_symbol=market_data_symbol,
            market_data_enabled=market_data_enabled,
        )
        if df.empty or "Close" not in df.columns:
            return {"momentum_score": 0.5, "quality_score": 0.5, "volatility_score": 0.5}

        closes = df["Close"].dropna()
        returns = closes.pct_change().dropna()
        ret_21 = float(closes.pct_change(21).iloc[-1]) if len(closes) > 21 else 0.0
        ret_63 = float(closes.pct_change(63).iloc[-1]) if len(closes) > 63 else ret_21
        ret_126 = float(closes.pct_change(126).iloc[-1]) if len(closes) > 126 else ret_63
        annualized_vol = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.25
        drawdown = self._drawdown(returns)

        momentum_score = float(np.clip(0.5 + ret_21 * 1.5 + ret_63 * 0.9 + ret_126 * 0.5, 0.05, 0.95))
        quality_score = float(np.clip(0.72 - abs(drawdown) * 0.9 - max(annualized_vol - 0.25, 0) * 0.35, 0.05, 0.95))
        volatility_score = float(np.clip(0.88 - annualized_vol * 1.1, 0.05, 0.95))

        return {
            "momentum_score": momentum_score,
            "quality_score": quality_score,
            "volatility_score": volatility_score,
        }

    def _factor_score(
        self,
        symbol: str,
        market_data_symbol: str | None = None,
        market_data_enabled: bool = True,
    ) -> float:
        metrics = self._factor_metrics(
            symbol,
            market_data_symbol=market_data_symbol,
            market_data_enabled=market_data_enabled,
        )
        return float(
            metrics["momentum_score"] * 0.45
            + metrics["quality_score"] * 0.35
            + metrics["volatility_score"] * 0.20
        )

    @staticmethod
    def _align_series(series: pd.Series, benchmark: pd.Series) -> pd.Series:
        if series.empty or benchmark.empty:
            return benchmark
        aligned = pd.concat([series, benchmark], axis=1).dropna()
        if aligned.empty:
            return benchmark
        return aligned.iloc[:, 1]

    @staticmethod
    def _calculate_cvar(returns: pd.Series, confidence_level: float) -> float:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        cutoff = clean.quantile(1 - confidence_level)
        tail = clean[clean <= cutoff]
        if tail.empty:
            return float(cutoff)
        return float(tail.mean())

    @staticmethod
    def _correlation_risk(returns_df: pd.DataFrame) -> float:
        if returns_df.shape[1] < 2:
            return 0.0
        corr = returns_df.corr().abs()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        upper = corr.where(mask)
        values = upper.stack()
        if values.empty:
            return 0.0
        return float(values.mean())

    @staticmethod
    def _fallback_correlation_matrix(symbols: list[str]) -> dict[str, Any]:
        matrix = []
        for row_symbol in symbols:
            row = []
            for col_symbol in symbols:
                if row_symbol == col_symbol:
                    row.append(1.0)
                else:
                    row.append(0.45)
            matrix.append(row)
        return {
            "symbols": symbols,
            "matrix": matrix,
            "as_of": None,
            "methodology": "Fallback static correlation matrix used when insufficient market history is available.",
        }

    def _liquidity_score(
        self,
        symbol: str,
        market_data_symbol: str | None = None,
        market_data_enabled: bool = True,
    ) -> float:
        df = self._history(
            symbol,
            period="6mo",
            market_data_symbol=market_data_symbol,
            market_data_enabled=market_data_enabled,
        )
        if df.empty or "Volume" not in df.columns or "Close" not in df.columns:
            return 0.5
        dollar_volume = (df["Close"] * df["Volume"]).tail(20).replace([np.inf, -np.inf], np.nan).dropna()
        if dollar_volume.empty:
            return 0.5
        avg_dollar_volume = float(dollar_volume.mean())
        return float(np.clip(np.log10(avg_dollar_volume + 1) / 9, 0.05, 0.99))

    @staticmethod
    def _resolve_market_data_symbol(
        symbol: str,
        market_data_symbol: str | None,
        market_data_enabled: bool,
    ) -> str | None:
        if not market_data_enabled:
            return None
        return market_data_symbol or symbol

    @staticmethod
    def _position_market_data_symbol(position: Any) -> str | None:
        return getattr(position, "market_data_symbol", None) or getattr(position, "symbol", None)

    @staticmethod
    def _position_market_data_enabled(position: Any) -> bool:
        return bool(getattr(position, "market_data_enabled", True))

    @staticmethod
    def _drawdown(returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        equity_curve = (1 + returns).cumprod()
        peak = equity_curve.cummax()
        drawdown = (equity_curve / peak) - 1
        return float(drawdown.min()) if not drawdown.empty else 0.0

    @staticmethod
    def _latest_value(indicators: pd.DataFrame, column: str, default: float) -> float:
        if indicators is None or indicators.empty or column not in indicators.columns:
            return default
        series = indicators[column].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return default
        return float(series.iloc[-1])

    @staticmethod
    def _max_weight(positions: list[Any]) -> float:
        total_value = sum(max(float(position.market_value), 0.0) for position in positions)
        if total_value <= 0:
            return 0.0
        return max(float(position.market_value) / total_value for position in positions)

    @staticmethod
    def _market_health_score(
        sp500: pd.DataFrame,
        nasdaq: pd.DataFrame,
        vix: pd.DataFrame,
        rates: float,
    ) -> float:
        score = 50.0

        if not sp500.empty and "Close" in sp500.columns:
            closes = sp500["Close"].dropna()
            if len(closes) > 20:
                ret_20 = float(closes.iloc[-1] / closes.iloc[-21] - 1)
                ma_50 = float(closes.tail(50).mean())
                score += np.clip(ret_20 * 140, -12, 12)
                score += 6 if float(closes.iloc[-1]) > ma_50 else -6

        if not nasdaq.empty and "Close" in nasdaq.columns:
            closes = nasdaq["Close"].dropna()
            if len(closes) > 20:
                ret_20 = float(closes.iloc[-1] / closes.iloc[-21] - 1)
                score += np.clip(ret_20 * 110, -10, 10)

        if not vix.empty and "Close" in vix.columns:
            vix_last = float(vix["Close"].dropna().iloc[-1])
            score += np.clip((22 - vix_last) * 1.6, -14, 14)

        if rates > 4.75:
            score -= 7
        elif rates < 3.75:
            score += 4

        return float(np.clip(score, 20, 80))


quant_insights_service = QuantInsightsService()
