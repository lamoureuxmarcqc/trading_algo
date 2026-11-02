from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.ai import (
    AiBacktestRequest,
    AiBacktestResponse,
    AiBacktestSymbolResult,
    AiDiagnosticsResponse,
    AiExplanationResponse,
    AiPaperStageRequest,
    AiPaperStageSkip,
    AiPredictionBandsResponse,
    AiSizingRequest,
    AiSizingResponse,
    AiStalenessFlags,
)
from app.schemas.terminal import TradingAlgoSymbolAnalysis
from app.services.trading_algo_terminal import TradingAlgoTerminalService


class PaperTradingDisabledError(ValueError):
    """Raised when paper-trading staging is requested while disabled."""


class PaperTradingGuardrailError(ValueError):
    """Raised when a paper-trading request violates a hard API guardrail."""


class AiTradingService:
    """Deterministic AI analytics and paper-trading planning layer."""

    def __init__(self, terminal_service: TradingAlgoTerminalService | None = None) -> None:
        self.terminal = terminal_service or TradingAlgoTerminalService()

    def get_diagnostics(self, symbol: str, period: str = "1y") -> AiDiagnosticsResponse:
        analysis = self.terminal.analyze_symbol(symbol, period)
        diagnostics = analysis.diagnostics
        if diagnostics is None:
            raise ValueError(f"diagnostics unavailable for {symbol}")
        return AiDiagnosticsResponse(
            symbol=analysis.symbol,
            model_version=diagnostics.model_version,
            feature_set_version=diagnostics.feature_set_version,
            trained_at=diagnostics.trained_at,
            inference_at=diagnostics.inference_at,
            confidence_score=diagnostics.confidence_score,
            directional_accuracy_30d=diagnostics.directional_accuracy_30d,
            directional_accuracy_90d=diagnostics.directional_accuracy_90d,
            hit_rate=diagnostics.hit_rate,
            mae=diagnostics.mae,
            rmse=diagnostics.rmse,
            signal_sharpe_30d=diagnostics.signal_sharpe_30d,
            signal_sharpe_90d=diagnostics.signal_sharpe_90d,
            staleness=AiStalenessFlags(
                model_stale=diagnostics.model_stale,
                market_data_stale=diagnostics.market_data_stale,
                data_drift_detected=diagnostics.data_drift_detected,
                concept_drift_detected=diagnostics.concept_drift_detected,
                regime_mismatch=diagnostics.regime_mismatch,
            ),
            diagnostics=diagnostics,
        )

    def get_explanation(self, symbol: str, period: str = "1y") -> AiExplanationResponse:
        analysis = self.terminal.analyze_symbol(symbol, period)
        source = analysis.explanation_source or "heuristic_fallback"
        return AiExplanationResponse(
            symbol=analysis.symbol,
            source=source,
            explained_at=analysis.explained_at or datetime.now(timezone.utc).isoformat(),
            features=analysis.feature_importance,
            fallback_labeled=source == "heuristic_fallback",
        )

    def get_prediction_bands(self, symbol: str, period: str = "1y") -> AiPredictionBandsResponse:
        analysis = self.terminal.analyze_symbol(symbol, period)
        return AiPredictionBandsResponse(
            symbol=analysis.symbol,
            as_of=analysis.as_of,
            latest_price=analysis.latest_price,
            bands=analysis.prediction_bands,
        )

    def size_position(self, payload: AiSizingRequest) -> AiSizingResponse:
        analysis = self.terminal.analyze_symbol(payload.symbol, payload.period)
        if analysis.sizing is None:
            raise ValueError(f"sizing unavailable for {payload.symbol}")
        sizing = analysis.sizing.model_copy(
            update={
                "recommended_notional": round(payload.nav * min(analysis.sizing.recommended_weight, payload.max_position_weight), 2),
                "recommended_quantity": round(
                    (payload.nav * min(analysis.sizing.recommended_weight, payload.max_position_weight))
                    / max(analysis.latest_price or 1.0, 1e-6),
                    4,
                ),
                "risk_budget_consumed": round(
                    min(analysis.sizing.recommended_weight / payload.max_position_weight, 1.0),
                    4,
                ),
            }
        )
        return AiSizingResponse(
            symbol=analysis.symbol,
            sizing=sizing,
            warnings=self.terminal.warning_flags(analysis.diagnostics),
        )

    def run_backtest(self, payload: AiBacktestRequest) -> AiBacktestResponse:
        lookback_days = self.terminal.parse_lookback_days(payload.lookback)
        results: list[AiBacktestSymbolResult] = []
        errors: list[str] = []
        for raw_symbol in payload.symbols:
            symbol = raw_symbol.strip().upper()
            if not symbol:
                continue
            try:
                analysis = self.terminal.analyze_symbol(symbol, payload.period)
                summary = self.terminal.backtest_with_policy(
                    symbol,
                    period=payload.period,
                    lookback_days=lookback_days,
                    buy_probability_threshold=payload.buy_probability_threshold,
                    stop_loss=payload.stop_loss,
                    take_profit=payload.take_profit,
                    slippage_bps=payload.slippage_bps,
                    max_position_weight=payload.max_position_weight,
                )
                results.append(
                    AiBacktestSymbolResult(
                        symbol=symbol,
                        summary=summary,
                        warnings=self.terminal.warning_flags(analysis.diagnostics),
                    )
                )
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")
        return AiBacktestResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            lookback=payload.lookback,
            buy_probability_threshold=payload.buy_probability_threshold,
            results=results,
            errors=errors,
        )

    def plan_paper_stage(self, payload: AiPaperStageRequest) -> tuple[list[dict], list[AiPaperStageSkip], list[str]]:
        if not payload.paper_trading_enabled:
            raise PaperTradingDisabledError("Paper trading staging is disabled by default.")
        if payload.stop_loss <= 0:
            raise PaperTradingGuardrailError("A positive stop-loss is mandatory for paper-trading staging.")
        if payload.max_notional <= 0:
            raise PaperTradingGuardrailError("max_notional must be positive.")

        candidates: list[dict] = []
        skipped: list[AiPaperStageSkip] = []
        warnings: list[str] = []
        for raw_symbol in payload.symbols:
            symbol = raw_symbol.strip().upper()
            if not symbol:
                continue
            try:
                analysis = self.terminal.analyze_symbol(symbol, payload.period)
                candidate, skip, symbol_warnings = self._candidate_from_analysis(analysis, payload)
                warnings.extend(symbol_warnings)
                if skip is not None:
                    skipped.append(skip)
                    continue
                if candidate is not None:
                    candidates.append(candidate)
            except Exception as exc:
                skipped.append(AiPaperStageSkip(symbol=symbol, reason=str(exc), warnings=[]))
        return candidates, skipped, sorted(set(warnings))

    def _candidate_from_analysis(
        self,
        analysis: TradingAlgoSymbolAnalysis,
        payload: AiPaperStageRequest,
    ) -> tuple[dict | None, AiPaperStageSkip | None, list[str]]:
        warnings = self.terminal.warning_flags(analysis.diagnostics)
        buy_probability = analysis.sizing.buy_probability if analysis.sizing else 0.0
        if buy_probability < payload.buy_probability_threshold:
            return None, AiPaperStageSkip(
                symbol=analysis.symbol,
                reason=f"buy_probability {buy_probability:.2f} below threshold {payload.buy_probability_threshold:.2f}",
                warnings=warnings,
            ), warnings
        if analysis.latest_price in (None, 0):
            return None, AiPaperStageSkip(
                symbol=analysis.symbol,
                reason="latest price unavailable",
                warnings=warnings,
            ), warnings
        weight = min(analysis.sizing.recommended_weight if analysis.sizing else 0.0, payload.max_position_weight)
        notional = min(payload.nav * weight, payload.max_notional)
        if notional <= 0:
            return None, AiPaperStageSkip(
                symbol=analysis.symbol,
                reason="recommended notional is zero after guardrails",
                warnings=warnings,
            ), warnings
        quantity = round(notional / analysis.latest_price, 4)
        stop_price = round(analysis.latest_price * (1 - payload.stop_loss), 4)
        take_profit_price = (
            round(analysis.latest_price * (1 + payload.take_profit), 4) if payload.take_profit else None
        )
        return (
            {
                "symbol": analysis.symbol,
                "side": "BUY",
                "order_type": "limit",
                "quantity": quantity,
                "limit_price": round(analysis.latest_price, 4),
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "buy_probability": buy_probability,
                "notional": round(notional, 2),
                "warnings": warnings,
                "policy_id": payload.policy_id,
            },
            None,
            warnings,
        )
