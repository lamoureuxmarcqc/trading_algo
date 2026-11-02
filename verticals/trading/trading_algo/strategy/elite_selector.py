class EliteSelector:

    def select_top(self, scored_universe, top_n=15):
        """
        scored_universe = [(ticker, score), ...]
        """

        ranked = sorted(scored_universe, key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
