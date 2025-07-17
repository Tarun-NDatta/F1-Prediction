import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .base_model import BaseModel

class RidgeModel(BaseModel):
    def __init__(self, alphas=np.logspace(-3, 3, 50)):
        super().__init__()
        self.alphas = alphas

    def train(self, X_train, y_train):
        self.pipeline = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=self.alphas, cv=5)
        )
        self.pipeline.fit(X_train, y_train)

        # Store fitted model for inspection (e.g., feature importances)
        self.model = self.pipeline.named_steps['ridgecv']
        return self
