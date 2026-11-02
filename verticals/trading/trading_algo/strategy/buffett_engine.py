import numpy as np

class BuffettEngine:

    def __init__(self, max_positions=10):
        self.max_positions = max_positions

    def compute_score(self, ticker_data):
        """
        ticker_data = {
            'pnl_pct': float,
            'sharpe': float,
            'weight': float,
            'revenue_growth': float,
            'profit_margin': float
        }
        """

        score = 0

        # 1. Performance (Buffett ignore court terme mais on garde un signal)
        score += ticker_data.get('pnl_pct', 0) * 0.2

        # 2. Qualité (fondamental simplifié)
        score += ticker_data.get('profit_margin', 0) * 0.3
        score += ticker_data.get('revenue_growth', 0) * 0.2

        # 3. Risk-adjusted return
        score += ticker_data.get('sharpe', 0) * 0.3

        # 4. Pénalité concentration excessive
        score -= ticker_data.get('weight', 0) * 0.4

        return score

    def rank_positions(self, performance, allocation):
        scores = {}

        for ticker, data in performance['positions'].items():
            ticker_data = {
                'pnl_pct': data.get('unrealized_pnl_pct', 0),
                'sharpe': data.get('sharpe_ratio', 0),
                'weight': allocation.get(ticker, 0),
                'revenue_growth': data.get('revenue_growth', 0),
                'profit_margin': data.get('profit_margin', 0)
            }

            scores[ticker] = self.compute_score(ticker_data)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def generate_target_allocation(self, ranked):
        """
        Allocation concentrée type Buffett :
        top 5 = 60-70%
        top 10 = 90%
        """

        allocation = {}
        total_weight = 0

        for i, (ticker, score) in enumerate(ranked):
            if i < 5:
                weight = 0.15  # 15% top conviction
            elif i < 10:
                weight = 0.05
            else:
                weight = 0.0

            allocation[ticker] = weight
            total_weight += weight

        # normalisation
        if total_weight > 0:
            for k in allocation:
                allocation[k] /= total_weight

        return allocation

    def generate_actions(self, current_alloc, target_alloc):
        actions = []

        for ticker, target_w in target_alloc.items():
            current_w = current_alloc.get(ticker, 0)

            diff = target_w - current_w

            if diff > 0.05:
                actions.append((ticker, "STRONG BUY"))
            elif diff > 0.02:
                actions.append((ticker, "BUY"))
            elif diff < -0.05:
                actions.append((ticker, "SELL"))
            elif diff < -0.02:
                actions.append((ticker, "REDUCE"))
            else:
                actions.append((ticker, "HOLD"))

        return actions
