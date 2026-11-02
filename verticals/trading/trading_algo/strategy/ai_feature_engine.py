import numpy as np

class AIFeatureEngine:

    def build_features(self, price_series):
        """
        price_series: pandas Series
        """

        returns = price_series.pct_change().dropna()

        features = {
            "momentum_1m": returns[-20:].mean(),
            "momentum_3m": returns[-60:].mean(),
            "volatility": returns.std(),
            "drawdown": (price_series.iloc[-1] / price_series.max()) - 1
        }

        return features
