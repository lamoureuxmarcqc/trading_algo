class CrashDetector:

    def detect(self, features):

        if features["drawdown"] < -0.15 and features["volatility"] > 0.03:
            return True

        return False
