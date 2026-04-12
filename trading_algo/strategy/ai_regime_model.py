from sklearn.ensemble import RandomForestClassifier
import numpy as np

class AIRegimeModel:

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50)
        self.trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict_proba(self, features):
        if not self.trained:
            return {"bull": 0.5, "bear": 0.5}

        probs = self.model.predict_proba([features])[0]

        return {
            "bull": probs[1],
            "bear": probs[0]
        }
