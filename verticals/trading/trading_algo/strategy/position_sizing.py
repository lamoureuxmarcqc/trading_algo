class PositionSizingEngine:

    def size_positions(self, ranked, regime):
        allocation = {}

        for i, (ticker, score) in enumerate(ranked):

            if regime in ["RISK ON", "BULLISH"]:
                if i < 5:
                    w = 0.18
                elif i < 10:
                    w = 0.05
                else:
                    w = 0

            elif regime == "DEFENSIVE":
                if i < 5:
                    w = 0.12
                elif i < 10:
                    w = 0.04
                else:
                    w = 0

            else:  # RISK OFF
                if i < 3:
                    w = 0.10
                else:
                    w = 0

            allocation[ticker] = w

        # normalize
        total = sum(allocation.values())
        if total > 0:
            for k in allocation:
                allocation[k] /= total

        return allocation
