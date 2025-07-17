import joblib
import numpy as np
from django.core.management.base import BaseCommand
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prediction.data_prep.pipeline import F1DataPipeline

class Command(BaseCommand):
    help = 'Train and evaluate Ridge Regression baseline model with recent season weighting'
    
    def add_arguments(self, parser):
        parser.add_argument('--alpha', type=float, default=1.0,
                            help='Regularization strength for Ridge Regression')
        parser.add_argument('--save', type=str, default='ridge_baseline.pkl',
                            help='Filename to save the trained model')
        parser.add_argument('--recent-weight', type=float, default=3.0,
                            help='Weight multiplier for recent seasons')

    def handle(self, *args, **options):
        # Initialize data pipeline
        pipeline = F1DataPipeline(impute_strategy='median')
        
        try:
            self.stdout.write("Preparing training data...")
            # Get training data and years
            X_train, X_test, y_train, y_test, years_train, _ = pipeline.prepare_training_data(include_years=True)
            
            self.stdout.write(f"Training samples: {X_train.shape[0]}")
            self.stdout.write(f"Test samples: {X_test.shape[0]}")
            self.stdout.write(f"Features: {X_train.shape[1]}")
            
            # Calculate sample weights based on training years
            if len(years_train) > 0:
                min_year = min(years_train)
                weights_train = np.array([
                    np.exp(-0.5 * (max(years_train) - year))  # Exponential decay
                    for year in years_train
                ])
                weights_train = weights_train * options['recent_weight']
                self.stdout.write(f"Applying recent weighting (min_year={min_year}, weight_factor={options['recent_weight']})")
            else:
                weights_train = np.ones(len(y_train))
                self.stdout.write("No year information - using uniform weights")
            
            # Initialize and train model
            model = Ridge(alpha=options['alpha'], random_state=42)
            self.stdout.write(f"Training Ridge Regression (alpha={options['alpha']}) with recent weighting...")
            model.fit(X_train, y_train, sample_weight=weights_train)
            
            # Evaluate on test set
            self.stdout.write("Evaluating model...")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            self.stdout.write("\n=== Evaluation Metrics ===")
            self.stdout.write(f"MAE: {mae:.4f}")
            self.stdout.write(f"RMSE: {rmse:.4f}")
            self.stdout.write(f"R²: {r2:.4f}")
            
            # Save model
            model_filename = options['save']
            joblib.dump(model, model_filename)
            
            # Save feature names for prediction
            feature_names = pipeline.get_feature_names()
            feature_filename = model_filename.replace('.pkl', '_features.pkl')
            joblib.dump(feature_names, feature_filename)
            
            self.stdout.write(self.style.SUCCESS(
                f"Model saved to {model_filename}"
            ))
            self.stdout.write(self.style.SUCCESS(
                f"Feature names saved to {feature_filename}"
            ))
            
            # Feature importance analysis
            self.stdout.write("\n=== Top 10 Features ===")
            coefs = model.coef_
            sorted_idx = np.argsort(np.abs(coefs))[::-1]
            
            for i in sorted_idx[:10]:
                self.stdout.write(f"{feature_names[i]:<25}: {coefs[i]:.4f}")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))