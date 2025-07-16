import joblib
import os
from django.conf import settings

def save_model(model, name):
    """Save trained model to disk"""
    models_dir = os.path.join(settings.BASE_DIR, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.joblib")
    joblib.dump(model, path)
    return path

def load_model(name):
    """Load trained model from disk"""
    path = os.path.join(settings.BASE_DIR, 'saved_models', f"{name}.joblib")
    return joblib.load(path)

def get_feature_names():
    """Get the list of feature names used in the pipeline"""
    return F1DataPipeline().feature_columns