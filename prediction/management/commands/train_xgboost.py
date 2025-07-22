from django.core.management.base import BaseCommand
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import os
import warnings
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

from skopt import BayesSearchCV
from skopt.space import Real, Integer

from prediction.data_prep.pipeline import F1DataPipeline

warnings.filterwarnings('ignore', category=FutureWarning)

class Command(BaseCommand):
    help = "Train optimized Ridge + XGBoost stacking ensemble for F1 prediction with SHAP and Bayesian tuning"

    def handle(self, *args, **kwargs):
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        model_prefix = f"f1_{version}"
        os.makedirs("models", exist_ok=True)

        self.stdout.write(self.style.NOTICE("🚦 Preparing data..."))
        pipeline = F1DataPipeline()
        # --- Enhanced Feature Integration ---
        df = pipeline.load_data()
        # Try to add enhanced features if all required columns exist and are not mostly NaN
        enhanced_cols = [
            'position_last_race', 'position_2races_ago', 'position_3races_ago',
            'driver_id', 'circuit', 'position',
            'soft_tire_laptime', 'medium_tire_laptime',
            'team', 'year', 'round', 'qualifying_time'
        ]
        if all(col in df.columns for col in enhanced_cols):
            # Check for mostly-NaN columns
            if df[enhanced_cols].isnull().mean().max() < 0.5:
                self.stdout.write(self.style.NOTICE("✨ Adding enhanced features..."))
                df = pipeline.create_enhanced_features(df)
            else:
                self.stdout.write(self.style.WARNING("Enhanced features skipped: too many missing values."))
        else:
            self.stdout.write(self.style.WARNING("Enhanced features skipped: required columns missing."))
        # Continue with missing value handling and splitting
        imputer = None
        # Only keep numeric columns for modeling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'position' in numeric_cols:
            numeric_cols.remove('position')
        # Impute if needed
        if df[numeric_cols].isnull().sum().any():
            self.stdout.write(self.style.WARNING("Missing values detected – applying median imputation"))
            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(df[numeric_cols])
        else:
            X = df[numeric_cols].values
        y = df['position'].values
        feature_names = numeric_cols
        # Use stratified split by position
        from sklearn.model_selection import train_test_split
        bins = pd.qcut(y, q=5, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=bins
        )
        # Apply position weights
        self.stdout.write("⚖️ Applying position-based weights...")
        def get_weights(y):
            w = np.ones_like(y)
            w[y <= 3] = 1.5
            w[(y > 3) & (y <= 10)] = 1.2
            w[y > 10] = 0.8
            return w
        sample_weights = get_weights(y_train)
        # Base Ridge
        ridge = Ridge(alpha=0.7, random_state=42)
        # --- Expanded Bayesian tuning for XGBoost ---
        self.stdout.write("🔬 Tuning XGBoost with Expanded Bayesian Optimization...")
        search = BayesSearchCV(
            estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, eval_metric='rmse'),
            search_spaces={
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.005, 0.2, prior='log-uniform'),
                'n_estimators': Integer(100, 1000),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0),
                'reg_alpha': Real(0.0, 5.0),
                'reg_lambda': Real(0.0, 5.0),
                'min_child_weight': Integer(1, 20),
                'gamma': Real(0.0, 2.0)
            },
            n_iter=60,  # More iterations for thorough search
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train, sample_weight=sample_weights)
        xgb_best = search.best_estimator_
        self.stdout.write(self.style.SUCCESS(f"🏆 Best XGBoost params: {search.best_params_}"))
        # Meta-learner
        meta = Ridge(alpha=0.1, random_state=42)
        # Ensemble
        self.stdout.write("🧠 Training Ridge + XGBoost stacking ensemble...")
        model = StackingRegressor(
            estimators=[("ridge", ridge), ("xgb", xgb_best)],
            final_estimator=meta,
            passthrough=False,
            cv=5,
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        # Evaluation
        self.stdout.write("\n📊 Model Performance")
        def print_metrics(y_true, y_pred, label):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            acc_3 = np.mean(np.abs(y_true - y_pred) <= 3)
            top3 = np.mean(y_pred[y_true <= 3] <= 3)
            points = np.mean(y_true[y_pred <= 10] <= 10)
            self.stdout.write(f"{label:<6} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f} | ±3 Acc: {acc_3:.3f} | Top3: {top3:.3f} | Points Acc: {points:.3f}")
        print_metrics(y_train, model.predict(X_train), "Train")
        print_metrics(y_test, model.predict(X_test), "Test")
        # Residuals
        residuals = y_test - model.predict(X_test)
        self.stdout.write("\n📉 Residual Bias by Position Range:")
        for label, mask in {
            "Podium": (y_test <= 3),
            "Points": ((y_test > 3) & (y_test <= 10)),
            "Back": (y_test > 10)
        }.items():
            if mask.any():
                self.stdout.write(f"{label:<8}: Bias = {residuals[mask].mean():+.3f} ± {residuals[mask].std():.3f}")
        # SHAP Feature Importance
        self.stdout.write("\n📈 SHAP Feature Importance (Top 10)")
        explainer = shap.Explainer(xgb_best, X_train, feature_names=feature_names)
        shap_values = explainer(X_train[:300])
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        shap_path = f"models/{model_prefix}_shap.png"
        plt.savefig(shap_path)
        plt.close()
        self.stdout.write(f"SHAP summary plot saved to {shap_path}")
        # Save artifacts
        self.stdout.write("\n💾 Saving model artifacts...")
        joblib.dump(model, f"models/{model_prefix}_stacked_model.pkl")
        joblib.dump(xgb_best, f"models/{model_prefix}_xgboost.pkl")
        joblib.dump(ridge, f"models/{model_prefix}_ridge.pkl")
        joblib.dump(feature_names, f"models/{model_prefix}_features.pkl")
        joblib.dump(pipeline.get_preprocessing_pipeline(), f"models/{model_prefix}_preprocessor.pkl")
        if imputer:
            joblib.dump(imputer, f"models/{model_prefix}_imputer.pkl")
        self.stdout.write(self.style.SUCCESS(f"\n✅ Model training complete: {model_prefix}"))
