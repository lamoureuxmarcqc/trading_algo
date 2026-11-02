import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, tickers):
        self.tickers = tickers

    def load_financials(self):
        data = {}
        for ticker in self.tickers:
            t = yf.Ticker(ticker)

            try:
                data[ticker] = {
                    "income": t.financials,
                    "balance": t.balance_sheet,
                    "cashflow": t.cashflow
                }
            except Exception as e:
                print(f"Erreur pour {ticker}: {e}")

        return data
    
    import pandas as pd

class REITMetrics:

    @staticmethod
    def compute_ffo(income, cashflow):
        try:
            ni = income.loc["Net Income"].iloc[0]
            dep = cashflow.loc["Depreciation"].iloc[0]
            gains = income.loc.get("Gain On Sale Of Assets", pd.Series([0])).iloc[0]
            return ni + dep - gains
        except:
            return None

    @staticmethod
    def compute_affo(ffo, cashflow):
        try:
            capex = cashflow.loc.get("Capital Expenditure", pd.Series([0])).iloc[0]
            return ffo - abs(capex)
        except:
            return None

    @staticmethod
    def ffo_per_share(ffo, shares_outstanding):
        try:
            return ffo / shares_outstanding
        except:
            return None

    @staticmethod
    def p_ffo(price, ffo_per_share):
        try:
            return price / ffo_per_share
        except:
            return None

    @staticmethod
    def debt_ratio(balance):
        try:
            return balance.loc["Total Debt"].iloc[0] / balance.loc["Total Assets"].iloc[0]
        except:
            return None

    @staticmethod
    def capital_deployment(balance):
        try:
            assets = balance.loc["Total Assets"]
            return (assets.iloc[0] - assets.iloc[1]) / assets.iloc[1]
        except:
            return None

    @staticmethod
    def acquisition_pipeline(cashflow):
        """
        Proxy simple :
        cash utilisé en investissement / total assets
        """
        try:
            investing = cashflow.loc["Investing Cash Flow"].iloc[0]
            return abs(investing)
        except:
            return None
class REITScoring:

    @staticmethod
    def score_value(p_ffo):
        if p_ffo is None:
            return 0
        if p_ffo < 12:
            return 5
        elif p_ffo < 18:
            return 4
        elif p_ffo < 25:
            return 3
        return 1

    @staticmethod
    def score_affo(affo):
        if affo is None:
            return 0
        if affo > 1e9:
            return 5
        elif affo > 5e8:
            return 4
        elif affo > 1e8:
            return 3
        return 1

    @staticmethod
    def score_pipeline(pipeline):
        if pipeline is None:
            return 0
        if pipeline > 1e9:
            return 5
        elif pipeline > 5e8:
            return 4
        elif pipeline > 1e8:
            return 3
        return 1

    @staticmethod
    def total_score(metrics):
        return sum(metrics.values())
from metrics import REITMetrics
from scoring import REITScoring
import yfinance as yf

class REITStrategy:

    def __init__(self, data):
        self.data = data

    def evaluate(self):
        results = {}

        for ticker, d in self.data.items():
            t = yf.Ticker(ticker)
            info = t.info

            shares = info.get("sharesOutstanding", None)
            price = info.get("currentPrice", None)

            ffo = REITMetrics.compute_ffo(d["income"], d["cashflow"])
            affo = REITMetrics.compute_affo(ffo, d["cashflow"])
            ffo_ps = REITMetrics.ffo_per_share(ffo, shares)
            pffo = REITMetrics.p_ffo(price, ffo_ps)
            debt = REITMetrics.debt_ratio(d["balance"])
            growth = REITMetrics.capital_deployment(d["balance"])
            pipeline = REITMetrics.acquisition_pipeline(d["cashflow"])

            scores = {
                "value": REITScoring.score_value(pffo),
                "affo": REITScoring.score_affo(affo),
                "pipeline": REITScoring.score_pipeline(pipeline)
            }

            total = REITScoring.total_score(scores)

            results[ticker] = {
                "FFO": ffo,
                "AFFO": affo,
                "FFO/share": ffo_ps,
                "P/FFO": pffo,
                "Debt": debt,
                "Growth": growth,
                "Pipeline": pipeline,
                "Score": total
            }

        return results
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class REITMLModel:

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    def train(self, df):
        """
        df doit contenir :
        - features (metrics)
        - target = performance future (ex: rendement 1 an)
        """
        X = df.drop(columns=["target"])
        y = df["target"]

        self.model.fit(X, y)

    def predict(self, df):
        return self.model.predict(df)

from data_loader import DataLoader
from strategy import REITStrategy
import pandas as pd

if __name__ == "__main__":

    tickers = ["O", "PLD", "SPG", "WELL", "VNQ"]

    loader = DataLoader(tickers)
    data = loader.load_financials()

    strategy = REITStrategy(data)
    results = strategy.evaluate()

    df = pd.DataFrame(results).T
    df = df.sort_values("Score", ascending=False)

    print(df)
    from ml_model import REITMLModel

# après création du dataframe df

features = df[[
    "P/FFO", "AFFO", "Debt", "Growth", "Pipeline"
]].fillna(0)

model = REITMLModel()

# Exemple fictif (à remplacer par données historiques réelles)
df["target"] = df["FFO"].pct_change().fillna(0)

model.train(df.dropna())

df["ML Score"] = model.predict(features)
df = df.sort_values("ML Score", ascending=False)

if __name__ == "__main__":
    sys.exit(int(main() or 0))