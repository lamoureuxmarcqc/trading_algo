class AIAllocationEngine:

    def allocate(self, ranked, regime_probs):

        bull = regime_probs["bull"]
        bear = regime_probs["bear"]

        allocation = {}

        for i, (ticker, score) in enumerate(ranked):

            base_weight = max(score, 0) / 100

            # amplification si bull
            weight = base_weight * (1 + bull)

            # réduction si bear
            weight *= (1 - bear)

            # concentration contrôlée
            if i < 5:
                weight *= 1.5

            allocation[ticker] = weight

        # normalize
        total = sum(allocation.values())
        if total > 0:
            for k in allocation:
                allocation[k] /= total

        return allocation