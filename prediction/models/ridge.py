import numpy as np
from sklearn.linear_model import RidgeCV
from .base_model import BaseModel

class RidgeModel(BaseModel):
    def __init__(self, alphas=np.logspace(-3, 3, 50)):
        super().__init__()
        self.alphas = alphas
        
    def train(self, X_train, y_train):
        from sklearn.pipeline import make_pipeline
        self.pipeline = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=self.alphas, cv=5)
        )
        self.pipeline.fit(X_train, y_train)
        return self