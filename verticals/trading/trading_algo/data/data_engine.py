import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class DataEngine:

    def __init__(self, extractor_class):
        self.extractor_class = extractor_class
        self.cache = {}

    def get_prices(self, tickers):
        extractor = self.extractor_class()

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(
                lambda t: (t, extractor.get_historical_data(t, "1d")),
                tickers
            ))

        prices = {}
        for t, df in results:
            if df is not None and not df.empty:
                prices[t] = float(df["Close"].iloc[-1])

        return prices