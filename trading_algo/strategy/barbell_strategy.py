from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BarbellConfig:
    defensive_weight_target: float | None = None
    opportunistic_weight_target: float | None = None
    cash_buffer_target: float | None = None
    max_positions_per_bucket: int = 3
    min_rebalance_delta: float = 0.01


@dataclass(frozen=True)
class BarbellCandidate:
    symbol: str
    bucket: str
    role: str
    current_weight: float
    buy_probability: float
    expected_return: float
    confidence_score: float
    quality_score: float
    volatility_score: float
    liquidity_score: float
    rationale: str


class BarbellStrategyEngine:
    def resolve_bucket_targets(self, regime: str, config: BarbellConfig) -> dict[str, float]:
        defaults = self._default_targets(regime)
        targets = {
            "defensive": (
                config.defensive_weight_target
                if config.defensive_weight_target is not None
                else defaults["defensive"]
            ),
            "opportunistic": (
                config.opportunistic_weight_target
                if config.opportunistic_weight_target is not None
                else defaults["opportunistic"]
            ),
            "cash": (
                config.cash_buffer_target
                if config.cash_buffer_target is not None
                else defaults["cash"]
            ),
        }
        total = sum(targets.values())
        if total <= 0:
            return defaults
        return {bucket: weight / total for bucket, weight in targets.items()}

    def select_candidates(
        self,
        candidates: list[BarbellCandidate],
        bucket: str,
        max_positions: int,
    ) -> list[BarbellCandidate]:
        filtered = [candidate for candidate in candidates if candidate.bucket == bucket]
        ranked = sorted(filtered, key=self._candidate_score, reverse=True)
        return ranked[:max_positions]

    def allocate_bucket(
        self,
        candidates: list[BarbellCandidate],
        total_weight: float,
    ) -> dict[str, float]:
        if total_weight <= 0 or not candidates:
            return {}

        scored = [(candidate.symbol, max(self._candidate_score(candidate), 0.05)) for candidate in candidates]
        score_total = sum(score for _, score in scored)
        if score_total <= 0:
            equal_weight = total_weight / len(scored)
            return {symbol: equal_weight for symbol, _score in scored}
        return {symbol: total_weight * (score / score_total) for symbol, score in scored}

    @staticmethod
    def _default_targets(regime: str) -> dict[str, float]:
        if regime == "risk_on":
            return {"defensive": 0.35, "opportunistic": 0.55, "cash": 0.10}
        if regime in {"risk_off", "defensive"}:
            return {"defensive": 0.55, "opportunistic": 0.25, "cash": 0.20}
        return {"defensive": 0.45, "opportunistic": 0.40, "cash": 0.15}

    def _candidate_score(self, candidate: BarbellCandidate) -> float:
        normalized_return = max(0.0, min((candidate.expected_return + 0.05) / 0.20, 1.0))
        if candidate.bucket == "defensive":
            return (
                candidate.quality_score * 0.35
                + candidate.volatility_score * 0.25
                + candidate.liquidity_score * 0.15
                + candidate.confidence_score * 0.10
                + normalized_return * 0.15
            )
        return (
            candidate.buy_probability * 0.35
            + normalized_return * 0.25
            + candidate.confidence_score * 0.15
            + candidate.quality_score * 0.10
            + candidate.volatility_score * 0.10
            + candidate.liquidity_score * 0.05
        )
