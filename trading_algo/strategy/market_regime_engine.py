class MarketRegimeEngine:

    def compute_regime(self, macro, indices):
        """
        Inputs:
        - macro: inflation, rates, etc.
        - indices: SP500, NASDAQ trends
        """

        score = 0

        # Trend (simple proxy)
        sp500 = indices.get("^GSPC", {}).get("closes", [])
        if len(sp500) > 20 and sp500[-1] > sum(sp500[-20:]) / 20:
            score += 30

        # Volatility proxy (simplifié)
        if len(sp500) > 5:
            recent_vol = abs(sp500[-1] - sp500[-5]) / sp500[-5]
            if recent_vol < 0.02:
                score += 20
            else:
                score -= 20

        # Macro
        inflation = macro.get("inflation", 2)
        rates = macro.get("rates", 3)

        if inflation < 3:
            score += 20
        else:
            score -= 20

        if rates < 4:
            score += 20
        else:
            score -= 20

        # Regime classification
        if score >= 60:
            return "RISK ON", score
        elif score >= 40:
            return "BULLISH", score
        elif score >= 20:
            return "NEUTRAL", score
        elif score >= 0:
            return "DEFENSIVE", score
        else:
            return "RISK OFF", score
