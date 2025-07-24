import os
import numpy as np
import pandas as pd
import joblib
from django.core.management.base import BaseCommand
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import xgboost as xgb

# Updated import as per your context
from prediction.data_prep.pipeline import F1DataPipeline


class Command(BaseCommand):
    help = "Train Ridge + XGBoost stacking ensemble model for F1 data 2022-2025"

    def handle(self, *args, **kwargs):
        # *** Change this to your project root directory ***
        ROOT_DIR = r"C:\Users\tarun\diss\td188"
        MODEL_PREFIX = "ensemble_xgb_ridge"

        self.stdout.write("🚦 Loading and preparing data...")
        pipeline = F1DataPipeline()
        try:
            df = pipeline.load_data()
        except NotImplementedError:
            self.stderr.write("Data loading not implemented. Please implement load_data in F1DataPipeline.")
            return

        try:
            df = pipeline.create_enhanced_features(df)
            self.stdout.write("Enhanced features created successfully.")
        except Exception as e:
            self.stderr.write(f"Error during feature engineering: {e}")
            return

        if 'year' in df.columns:
            df = df[df['year'].between(2022, 2025)]
            self.stdout.write(f"Data filtered to years 2022-2025, {len(df)} rows remain.")
        else:
            self.stdout.write("Warning: 'year' column not found; skipping year filtering.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'position' in numeric_cols:
            numeric_cols.remove('position')
        X_raw = df[numeric_cols]
        y = df['position'].values

        preprocessing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        self.stdout.write("Imputing and scaling features...")
        X = preprocessing_pipeline.fit_transform(X_raw)

        y_bins = pd.qcut(y, 5, duplicates="drop")
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, stratify=y_bins, random_state=42)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_df = df.iloc[train_idx]

        base_year = 2025

        def compute_sample_weight(row):
            year_diff = base_year - row['year']
            return np.exp(-0.05 * year_diff) * 3.0

        sample_weights = train_df.apply(compute_sample_weight, axis=1).values

        self.stdout.write(f"Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

        self.stdout.write("Training Ridge regression model...")
        ridge = Ridge(alpha=0.7, random_state=42)
        ridge.fit(X_train, y_train, sample_weight=sample_weights)

        self.stdout.write("Starting Bayesian optimization for XGBoost...")

        xgb_estimator = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
            eval_metric='rmse'
        )

        param_spaces = {
            "n_estimators": Integer(100, 1000),
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.005, 0.2, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "reg_alpha": Real(0.0, 5.0),
            "reg_lambda": Real(0.0, 5.0),
            "min_child_weight": Integer(1, 20),
            "gamma": Real(0.0, 2.0),
        }

        bayes_search = BayesSearchCV(
            estimator=xgb_estimator,
            search_spaces=param_spaces,
            n_iter=40,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42
        )

        bayes_search.fit(X_train, y_train, sample_weight=sample_weights)

        best_xgb = bayes_search.best_estimator_
        self.stdout.write(f"Best XGBoost params: {bayes_search.best_params_}")

        self.stdout.write("Training stacking ensemble...")
        stacking_model = StackingRegressor(
            estimators=[("ridge", ridge), ("xgb", best_xgb)],
            final_estimator=Ridge(alpha=0.1, random_state=42),
            n_jobs=-1,
            cv=5,
            passthrough=False
        )
        stacking_model.fit(X_train, y_train, sample_weight=sample_weights)

        def evaluate_model(name, model, X_, y_):
            preds = model.predict(X_)
            rmse = np.sqrt(mean_squared_error(y_, preds))
            mae = mean_absolute_error(y_, preds)
            r2 = r2_score(y_, preds)
            self.stdout.write(f"{name} Performance: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        evaluate_model("Train", stacking_model, X_train, y_train)
        evaluate_model("Test", stacking_model, X_test, y_test)

        # Save models and preprocessing pipeline using absolute root directory paths
        ridge_path = os.path.join(ROOT_DIR, f"{MODEL_PREFIX}_ridge.pkl")
        xgb_path = os.path.join(ROOT_DIR, f"{MODEL_PREFIX}_xgboost.pkl")
        ensemble_path = os.path.join(ROOT_DIR, f"{MODEL_PREFIX}.pkl")
        features_path = os.path.join(ROOT_DIR, f"{MODEL_PREFIX}_features.pkl")
        preproc_path = os.path.join(ROOT_DIR, f"{MODEL_PREFIX}_preprocessing.pkl")

        joblib.dump(ridge, ridge_path)
        joblib.dump(best_xgb, xgb_path)
        joblib.dump(stacking_model, ensemble_path)
        joblib.dump(numeric_cols, features_path)
        joblib.dump(preprocessing_pipeline, preproc_path)

        self.stdout.write(f"Models and artifacts saved to {ROOT_DIR}")
        self.stdout.write("✅ Training complete.")