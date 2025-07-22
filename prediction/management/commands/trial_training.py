import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class Command(BaseCommand):
    help = "Train Ridge + XGBoost stacking ensemble"

    def add_arguments(self, parser):
        parser.add_argument('--output', type=str, default='stacking_model.pkl',
                          help='Output model file (default: stacking_model.pkl)')
        parser.add_argument('--test-size', type=float, default=0.2,
                          help='Test size for validation (default: 0.2)')

    def handle(self, *args, **options):
        try:
            # 1. Load and prepare data
            X, y = self.load_training_data()
            
            # 2. Create stacking ensemble
            stacking_model = self.create_stacking_model()
            
            # 3. Train with cross-validation
            self.train_and_evaluate(stacking_model, X, y)
            
            # 4. Save final model
            self.save_model(stacking_model, X, y, options['output'])
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Training failed: {str(e)}"))

    def load_training_data(self):
        """Load training data from your existing datasets"""
        # This is a placeholder - you'll need to adapt this to your data loading logic
        from data.models import RaceResult, DriverPerformance, TeamPerformance
        
        # Example data loading - adapt to your specific needs
        race_results = RaceResult.objects.all().select_related('driver', 'session__event')
        
        X = []
        y = []
        
        features = [
            'moving_avg_5', 'qualifying_avg', 'position_variance',
            'points_per_race', 'circuit_affinity', 'quali_improvement',
            'teammate_battle', 'wet_weather_perf', 'reliability_score',
            'quali_race_delta', 'position_momentum', 'dnf_rate',
            'pit_stop_avg'
        ]
        
        for result in race_results:
            try:
                driver_perf = DriverPerformance.objects.get(
                    driver=result.driver, 
                    event=result.session.event
                )
                team_perf = TeamPerformance.objects.get(
                    team=result.team,
                    event=result.session.event
                )
                
                row = []
                for feat in features:
                    val = 0.0
                    if hasattr(driver_perf, feat):
                        val = getattr(driver_perf, feat) or 0.0
                    elif hasattr(team_perf, feat):
                        val = getattr(team_perf, feat) or 0.0
                    row.append(float(val))
                
                X.append(row)
                y.append(result.position)
                
            except:
                continue
        
        self.stdout.write(self.style.SUCCESS(f"Loaded {len(X)} training samples"))
        return np.array(X), np.array(y)

    def create_stacking_model(self):
        """Create the stacking ensemble with Ridge and XGBoost"""
        
        # Base models
        ridge = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
        
        xgboost = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('xgb', XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        
        # Meta-learner (final model)
        meta_learner = Ridge(alpha=0.1)
        
        # Stacking ensemble
        stacking_model = StackingRegressor(
            estimators=[
                ('ridge', ridge),
                ('xgboost', xgboost)
            ],
            final_estimator=meta_learner,
            cv=5,  # 5-fold cross-validation to generate base predictions
            n_jobs=-1
        )
        
        return stacking_model

    def train_and_evaluate(self, model, X, y):
        """Train and evaluate the stacking model"""
        
        # Cross-validation evaluation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_mean_absolute_error'
        )
        
        self.stdout.write(self.style.SUCCESS(
            f"Cross-validation MAE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})"
        ))
        
        return model

    def save_model(self, model, X, y, output_path):
        """Train final model and save"""
        
        # Train on full dataset
        model.fit(X, y)
        
        # Save model
        joblib.dump(model, output_path)
        
        self.stdout.write(self.style.SUCCESS(f"Stacking model saved to {output_path}"))
        self.stdout.write(self.style.INFO(
            f"Base estimators: {[name for name, _ in model.estimators_]}"
        ))
        self.stdout.write(self.style.INFO(
            f"Meta-learner: {type(model.final_estimator_).__name__}"
        ))