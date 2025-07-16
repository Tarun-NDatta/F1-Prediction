# prediction/management/commands/train_baseline.py
import joblib
import numpy as np
from django.core.management.base import BaseCommand
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prediction.data_prep.pipeline import F1DataPipeline

class Command(BaseCommand):
    help = 'Train and evaluate Ridge Regression baseline model'
    
    def add_arguments(self, parser):
        parser.add_argument('--alpha', type=float, default=1.0,
                            help='Regularization strength for Ridge Regression')
        parser.add_argument('--save', type=str, default='ridge_baseline.pkl',
                            help='Filename to save the trained model')

    def handle(self, *args, **options):
        # Initialize data pipeline
        pipeline = F1DataPipeline(impute_strategy='median')
        
        try:
            self.stdout.write("Preparing training data...")
            X_train, X_test, y_train, y_test = pipeline.prepare_training_data()
            
            self.stdout.write(f"Training samples: {X_train.shape[0]}")
            self.stdout.write(f"Test samples: {X_test.shape[0]}")
            self.stdout.write(f"Features: {X_train.shape[1]}")
            
            # Initialize and train model
            model = Ridge(alpha=options['alpha'], random_state=42)
            self.stdout.write(f"Training Ridge Regression (alpha={options['alpha']})...")
            model.fit(X_train, y_train)
            
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
            joblib.dump(model, options['save'])
            self.stdout.write(self.style.SUCCESS(
                f"Model saved to {options['save']}"
            ))
            
            # Feature importance analysis
            self.stdout.write("\n=== Top 10 Features ===")
            features = pipeline.get_feature_names()
            coefs = model.coef_
            sorted_idx = np.argsort(np.abs(coefs))[::-1]
            
            for i in sorted_idx[:10]:
                self.stdout.write(f"{features[i]:<25}: {coefs[i]:.4f}")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))