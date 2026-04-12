class RiskOverlay:

    def apply(self, allocation, analysis):

        drawdown = analysis['performance'].get('total_pnl_pct', 0)

        # Hard risk control
        if drawdown < -15:
            return {k: v * 0.5 for k, v in allocation.items()}

        # concentration cap
        capped = {}
        for k, v in allocation.items():
            capped[k] = min(v, 0.25)

        return capped
