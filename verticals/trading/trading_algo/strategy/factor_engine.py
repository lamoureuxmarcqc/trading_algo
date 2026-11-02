class FactorEngine:

    def compute_quality_score(self, fundamentals):
        """
        fundamentals:
        {
            'roic': float,
            'gross_margin': float,
            'net_margin': float,
            'revenue_growth': float,
            'fcf_yield': float,
            'debt_to_equity': float
        }
        """

        score = 0

        # Profitability (Buffett core)
        score += fundamentals.get('roic', 0) * 0.25
        score += fundamentals.get('net_margin', 0) * 0.15
        score += fundamentals.get('gross_margin', 0) * 0.10

        # Growth (disciplined)
        score += fundamentals.get('revenue_growth', 0) * 0.15

        # Cash generation
        score += fundamentals.get('fcf_yield', 0) * 0.20

        # Balance sheet penalty
        debt = fundamentals.get('debt_to_equity', 1)
        score -= debt * 0.15

        return score
