from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prediction.data_prep.utilities import save_model, load_model

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.pipeline = None
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        return {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': mean_squared_error(y_test, predictions, squared=False)
        }
    
    def predict(self, X):
        """Make predictions"""
        return self.pipeline.predict(X)
    
    def save(self, name):
        """Save the entire pipeline"""
        return save_model(self.pipeline, name)
    
    @classmethod
    def load(cls, name):
        """Load a saved pipeline"""
        loaded = load_model(name)
        instance = cls()
        instance.pipeline = loaded
        return instance