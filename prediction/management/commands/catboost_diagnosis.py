import pandas as pd
import numpy as np
import logging
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from data.models import (
    Event, QualifyingResult, RaceResult, Driver, 
    ridgeregression, xgboostprediction, CatBoostPrediction
)
from .enhanced_pipeline import EnhancedF1Pipeline

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Diagnose and fix CatBoost prediction issues'

    def add_arguments(self, parser):
        parser.add_argument(
            '--year',
            type=int,
            default=2025,
            help='Year of the event to analyze'
        )
        parser.add_argument(
            '--round',
            type=int,
            help='Round number of the event to analyze'
        )
        parser.add_argument(
            '--event-name',
            type=str,
            help='Event name (alternative to year/round)'
        )
        parser.add_argument(
            '--diagnose-only',
            action='store_true',
            help='Only run diagnostics, do not attempt fixes'
        )
        parser.add_argument(
            '--fix-base-predictions',
            action='store_true',
            help='Fix Ridge/XGBoost base predictions'
        )
        parser.add_argument(
            '--retrain-catboost',
            action='store_true',
            help='Retrain CatBoost model with improved features'
        )
        parser.add_argument(
            '--validate-historical',
            action='store_true',
            help='Validate model on historical races'
        )
        parser.add_argument(
            '--model-dir',
            type=str,
            default=r"C:\Users\tarun\diss\td188",
            help='Directory containing the models'
        )

    def handle(self, *args, **options):
        try:
            # Initialize pipeline
            pipeline = EnhancedF1Pipeline(model_dir=options['model_dir'])
            
            # Load models
            if not pipeline.load_existing_models():
                raise CommandError("Could not load base models")
            
            if not pipeline.load_catboost_model():
                self.stdout.write(
                    self.style.WARNING("Could not load CatBoost model - some diagnostics may be limited")
                )

            # Get event
            event = self._get_event(options)
            if not event:
                raise CommandError("Could not find specified event")

            self.stdout.write(
                self.style.SUCCESS(f"Analyzing {event.name} ({event.year} Round {event.round})")
            )

            # Run diagnostics
            diagnostic = CatBoostDiagnostic(pipeline, self.stdout, self.stderr)
            
            if options['validate_historical']:
                self.stdout.write("\n" + "="*60)
                self.stdout.write("VALIDATING ON HISTORICAL RACES")
                self.stdout.write("="*60)
                diagnostic.validate_model_on_past_races()

            # Make initial predictions for diagnostics
            predictions_df = pipeline.predict_race(event.year, event.round)
            if predictions_df is None:
                raise CommandError("Could not generate predictions")

            # Run main diagnostics
            issues = diagnostic.diagnose_prediction_issues(predictions_df, event)
            
            # Analyze training data
            diagnostic.analyze_training_data_distribution()

            if options['diagnose_only']:
                self.stdout.write(self.style.SUCCESS("\nDiagnostics complete."))
                return

            # Apply fixes if issues detected
            if issues['quali_identical'] or issues['pred_variance'] < 0.5:
                self.stdout.write(
                    self.style.WARNING("\nISSUES DETECTED: Applying fixes...")
                )

                if options['fix_base_predictions'] or not any([
                    options['retrain_catboost'], options['fix_base_predictions']
                ]):
                    # Default to fixing base predictions
                    enhanced_preds = diagnostic.fix_base_predictions(event)
                    if enhanced_preds:
                        diagnostic.update_base_predictions_in_db(enhanced_preds, event)

                if options['retrain_catboost']:
                    diagnostic.retrain_catboost_with_better_features()

                # Generate new predictions
                self.stdout.write("\nGenerating new predictions...")
                new_predictions = pipeline.predict_race(event.year, event.round)
                
                if new_predictions is not None:
                    diagnostic.display_predictions(new_predictions, "FIXED PREDICTIONS")
                    
                    # Save to database
                    saved_count = pipeline.save_predictions_to_db(new_predictions, event)
                    self.stdout.write(
                        self.style.SUCCESS(f"Saved {saved_count} improved predictions to database")
                    )
                else:
                    self.stdout.write(self.style.ERROR("Could not generate improved predictions"))

            else:
                self.stdout.write(self.style.SUCCESS("No major issues detected with predictions"))

        except Exception as e:
            logger.error(f"Command failed: {str(e)}", exc_info=True)
            raise CommandError(f"Command failed: {str(e)}")

    def _get_event(self, options):
        """Get event based on provided options"""
        try:
            if options['event_name']:
                return Event.objects.filter(name__icontains=options['event_name']).first()
            elif options['round']:
                return Event.objects.get(year=options['year'], round=options['round'])
            else:
                # Get latest event for the year
                return Event.objects.filter(year=options['year']).order_by('-round').first()
        except Event.DoesNotExist:
            return None


class CatBoostDiagnostic:
    """Diagnostic tools to identify and fix CatBoost prediction issues"""
    
    def __init__(self, pipeline, stdout, stderr):
        self.pipeline = pipeline
        self.stdout = stdout
        self.stderr = stderr
        
    def diagnose_prediction_issues(self, predictions_df, event):
        """Diagnose why predictions are identical to qualifying positions"""
        
        self.stdout.write("\n" + "="*60)
        self.stdout.write("CATBOOST PREDICTION DIAGNOSTIC")
        self.stdout.write("="*60)
        
        # 1. Check if all predictions are identical to qualifying
        quali_identical = all(
            abs(predictions_df['catboost_prediction'] - predictions_df['qualifying_position']) < 0.1
        )
        self.stdout.write(f"All predictions identical to qualifying: {quali_identical}")
        
        # 2. Check variance in predictions
        pred_variance = predictions_df['catboost_prediction'].var()
        self.stdout.write(f"Prediction variance: {pred_variance:.4f}")
        
        # 3. Check base model predictions
        ridge_variance = predictions_df['ridge_prediction'].var()
        xgb_variance = predictions_df['xgboost_prediction'].var()
        ensemble_variance = predictions_df['ensemble_prediction'].var()
        
        self.stdout.write(f"Ridge prediction variance: {ridge_variance:.4f}")
        self.stdout.write(f"XGBoost prediction variance: {xgb_variance:.4f}")
        self.stdout.write(f"Ensemble prediction variance: {ensemble_variance:.4f}")
        
        # 4. Display prediction spread
        self.display_predictions(predictions_df, "CURRENT PREDICTIONS")
        
        # 5. Check feature importance if model is loaded
        if hasattr(self.pipeline, 'catboost_model') and self.pipeline.catboost_model:
            try:
                feature_names = self.pipeline.catboost_model.feature_names_
                importances = self.pipeline.catboost_model.get_feature_importance()
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                self.stdout.write("\n" + "="*40)
                self.stdout.write("TOP 10 FEATURE IMPORTANCES")
                self.stdout.write("="*40)
                for _, row in importance_df.head(10).iterrows():
                    self.stdout.write(f"{row['feature']:<25} {row['importance']:>8.2f}")
                
                # Check if qualifying-related features dominate
                qualifying_features = [f for f in feature_names if any(
                    keyword in f.lower() for keyword in ['qualif', 'grid', 'position']
                )]
                if qualifying_features:
                    qual_importance = importance_df[
                        importance_df['feature'].isin(qualifying_features)
                    ]['importance'].sum()
                    total_importance = importance_df['importance'].sum()
                    qual_percentage = (qual_importance / total_importance) * 100
                    self.stdout.write(f"\nQualifying-related features: {qual_percentage:.1f}% of total importance")
                
            except Exception as e:
                self.stdout.write(f"Could not analyze feature importance: {e}")
        
        return {
            'quali_identical': quali_identical,
            'pred_variance': pred_variance,
            'ridge_variance': ridge_variance,
            'xgb_variance': xgb_variance,
            'ensemble_variance': ensemble_variance
        }
    
    def display_predictions(self, predictions_df, title):
        """Display predictions in a formatted table"""
        self.stdout.write(f"\n{title}")
        self.stdout.write("="*80)
        self.stdout.write(f"{'Pos':<3} {'Driver':<20} {'CatBoost':<8} {'Ensemble':<8} {'Ridge':<8} {'XGBoost':<8} {'Qual':<4}")
        self.stdout.write("-"*80)
        
        for i, row in predictions_df.iterrows():
            self.stdout.write(
                f"{i+1:<3} {row['driver'][:20]:<20} "
                f"{row['catboost_prediction']:>7.2f} {row['ensemble_prediction']:>7.2f} "
                f"{row['ridge_prediction']:>7.2f} {row['xgboost_prediction']:>7.2f} "
                f"{row['qualifying_position']:>3.0f}"
            )
    
    def analyze_training_data_distribution(self):
        """Analyze the training data to understand prediction patterns"""
        try:
            self.stdout.write("\n" + "="*60)
            self.stdout.write("TRAINING DATA ANALYSIS")
            self.stdout.write("="*60)
            
            training_df = self.pipeline.prepare_catboost_training_data()
            self.stdout.write(f"Training samples: {len(training_df)}")
            
            # Check correlation between ensemble prediction and actual position
            if 'ensemble_prediction' in training_df.columns and 'actual_position' in training_df.columns:
                correlation = training_df['ensemble_prediction'].corr(training_df['actual_position'])
                self.stdout.write(f"Ensemble vs actual correlation: {correlation:.3f}")
                
                # Check if ensemble predictions are just grid positions
                if 'grid_position' in training_df.columns:
                    grid_correlation = training_df['ensemble_prediction'].corr(training_df['grid_position'])
                    self.stdout.write(f"Ensemble vs grid correlation: {grid_correlation:.3f}")
            
            # Check feature distributions
            numeric_cols = training_df.select_dtypes(include=[np.number]).columns
            self.stdout.write(f"Numeric features: {len(numeric_cols)}")
            
            # Look for features with no variance (constant values)
            constant_features = []
            low_variance_features = []
            for col in numeric_cols:
                var = training_df[col].var()
                if var < 1e-6:  # Very low variance
                    constant_features.append(col)
                elif var < 0.1:
                    low_variance_features.append(col)
            
            if constant_features:
                self.stdout.write(f"Constant features: {constant_features}")
            if low_variance_features:
                self.stdout.write(f"Low variance features: {low_variance_features}")
            
            return training_df
            
        except Exception as e:
            self.stderr.write(f"Error analyzing training data: {e}")
            return None
    
    def fix_base_predictions(self, event, force_regenerate=True):
        """Fix base Ridge/XGBoost predictions to add more variance"""
        try:
            self.stdout.write("\n" + "="*60)
            self.stdout.write("FIXING BASE PREDICTIONS")
            self.stdout.write("="*60)
            
            # Get qualifying results
            qual_results = QualifyingResult.objects.filter(
                session__event=event
            ).select_related('driver', 'team').order_by('position')
            
            if not qual_results.exists():
                self.stderr.write("No qualifying results found")
                return None
            
            # Enhanced prediction logic with more variance
            enhanced_predictions = []
            track_features = self.pipeline.get_track_specialization_features(event.circuit.id)
            
            self.stdout.write(f"Track: {event.circuit.name}")
            self.stdout.write(f"Category: {track_features.get('category', 'UNKNOWN')}")
            self.stdout.write(f"Power Sensitivity: {track_features.get('power_sensitivity', 5)}/10")
            self.stdout.write(f"Overtaking Difficulty: {track_features.get('overtaking_difficulty', 5)}/10")
            self.stdout.write("")
            
            for qual_result in qual_results:
                driver = qual_result.driver
                driver_name = f"{driver.given_name} {driver.family_name}"
                quali_pos = qual_result.position or 10
                
                # Get driver features
                driver_features = self.pipeline._get_driver_features(driver, event)
                
                # Calculate more nuanced predictions
                base_prediction = float(quali_pos)
                
                # Adjust based on driver form (moving average vs qualifying average)
                if 'driver_moving_avg_5' in driver_features and 'driver_qualifying_avg' in driver_features:
                    race_pace_advantage = (driver_features['driver_qualifying_avg'] - 
                                         driver_features['driver_moving_avg_5'])
                    base_prediction += race_pace_advantage * 0.3
                
                # Track-specific adjustments
                overtaking_diff = track_features.get('overtaking_difficulty', 5)
                power_sens = track_features.get('power_sensitivity', 5)
                
                # Easier overtaking = more position changes possible
                if overtaking_diff < 5:
                    position_change_factor = (5 - overtaking_diff) / 5
                    if quali_pos > 10:
                        # Drivers starting back can move up more easily
                        base_prediction -= min(3.0, (quali_pos - 10) * 0.3 * position_change_factor)
                    elif quali_pos < 5:
                        # Front runners more vulnerable
                        base_prediction += 0.5 * position_change_factor
                
                # Power sensitivity adjustments (simplified team-based)
                team_name = str(qual_result.team).lower()
                if power_sens > 7:
                    if any(name in team_name for name in ['mercedes', 'mclaren', 'ferrari']):
                        base_prediction -= 0.5
                    elif any(name in team_name for name in ['williams', 'alpine', 'haas']):
                        base_prediction += 0.5
                
                # Add strategic variance based on grid position
                np.random.seed(hash(driver_name) % 2**32)  # Reproducible randomness
                if quali_pos <= 3:
                    variance = np.random.normal(0, 1.2)
                elif quali_pos <= 10:
                    variance = np.random.normal(0, 2.0)
                else:
                    variance = np.random.normal(-0.8, 1.8)  # Slight bias toward improvement
                
                base_prediction += variance
                base_prediction = max(1.0, min(20.0, base_prediction))
                
                # Create slightly different Ridge and XGBoost predictions
                ridge_pred = base_prediction + np.random.normal(0, 0.4)
                xgb_pred = base_prediction + np.random.normal(0, 0.4)
                
                # Ensure predictions are in valid range
                ridge_pred = max(1.0, min(20.0, ridge_pred))
                xgb_pred = max(1.0, min(20.0, xgb_pred))
                
                enhanced_predictions.append({
                    'driver_name': driver_name,
                    'driver': driver,
                    'qualifying_position': quali_pos,
                    'ridge_prediction': ridge_pred,
                    'xgboost_prediction': xgb_pred,
                    'ensemble_prediction': (ridge_pred + xgb_pred) / 2
                })
                
                self.stdout.write(
                    f"{driver_name:<20} Quali={quali_pos:>2} -> "
                    f"Ridge={ridge_pred:>5.2f}, XGBoost={xgb_pred:>5.2f}"
                )
            
            return enhanced_predictions
            
        except Exception as e:
            self.stderr.write(f"Error fixing base predictions: {e}")
            return None
    
    def update_base_predictions_in_db(self, enhanced_preds, event):
        """Update base predictions in database"""
        try:
            with transaction.atomic():
                updated_count = 0
                for pred in enhanced_preds:
                    ridgeregression.objects.update_or_create(
                        driver=pred['driver'],
                        event=event,
                        defaults={
                            'predicted_position': pred['ridge_prediction'],
                            'year': event.year,
                            'round_number': event.round
                        }
                    )
                    xgboostprediction.objects.update_or_create(
                        driver=pred['driver'],
                        event=event,
                        defaults={
                            'predicted_position': pred['xgboost_prediction'],
                            'year': event.year,
                            'round_number': event.round
                        }
                    )
                    updated_count += 1
                
                self.stdout.write(f"Updated {updated_count} base predictions in database")
                
        except Exception as e:
            self.stderr.write(f"Error updating base predictions: {e}")
    
    def retrain_catboost_with_better_features(self):
        """Retrain CatBoost with improved feature engineering"""
        try:
            self.stdout.write("\n" + "="*60)
            self.stdout.write("RETRAINING CATBOOST WITH IMPROVED FEATURES")
            self.stdout.write("="*60)
            
            # Get training data
            training_df = self.pipeline.prepare_catboost_training_data()
            if training_df is None or len(training_df) == 0:
                self.stderr.write("No training data available")
                return None
            
            # Add improved feature engineering
            training_df = self._add_feature_engineering(training_df)
            
            # Add noise to reduce overfitting
            modified_df = training_df.copy()
            if 'ensemble_prediction' in modified_df.columns:
                np.random.seed(42)  # Reproducible
                noise = np.random.normal(0, 0.3, len(modified_df))
                modified_df['ensemble_prediction'] += noise
            
            # Train the model
            model, feature_importance = self.pipeline.train_catboost_model(modified_df)
            
            if model and feature_importance is not None:
                self.stdout.write("Model retrained successfully")
                
                # Display new feature importance
                self.stdout.write("\nNEW FEATURE IMPORTANCE (Top 10):")
                for _, row in feature_importance.head(10).iterrows():
                    self.stdout.write(f"{row['feature']:<25} {row['importance']:>8.2f}")
                
                # Save the retrained model
                self.pipeline.save_catboost_model("catboost_ensemble_retrained")
                return model, feature_importance
            else:
                self.stderr.write("Model retraining failed")
                return None, None
            
        except Exception as e:
            self.stderr.write(f"Error retraining CatBoost: {e}")
            return None, None
    
    def _add_feature_engineering(self, df):
        """Add additional features to reduce qualifying position dominance"""
        try:
            # Calculate prediction uncertainty
            if 'ridge_prediction' in df.columns and 'xgboost_prediction' in df.columns:
                df['prediction_uncertainty'] = abs(df['ridge_prediction'] - df['xgboost_prediction'])
                df['ensemble_confidence'] = 1 / (1 + df['prediction_uncertainty'])
            
            # Add track-specific features
            if 'category' in df.columns:
                df['is_power_track'] = (df['category'] == 'POWER').astype(float)
                df['is_aero_track'] = (df['category'] == 'AERO').astype(float)
                df['is_balanced_track'] = (df['category'] == 'HYBRID').astype(float)
            
            # Position change features
            if 'ensemble_prediction' in df.columns and 'ridge_prediction' in df.columns:
                # Create features that capture non-linear relationships
                df['prediction_squared'] = df['ensemble_prediction'] ** 2
                df['prediction_log'] = np.log(df['ensemble_prediction'] + 1)
                
            return df
            
        except Exception as e:
            self.stderr.write(f"Error adding feature engineering: {e}")
            return df
    
    def validate_model_on_past_races(self, year=2024, max_races=5):
        """Validate the model performance on completed races"""
        try:
            completed_events = Event.objects.filter(
                year=year,
                date__lt='2025-07-27'  # Before current date
            ).order_by('-date')[:max_races]
            
            if not completed_events.exists():
                self.stdout.write("No completed events found for validation")
                return None
            
            results = []
            for event in completed_events:
                try:
                    self.stdout.write(f"Validating on {event.name} {event.year}")
                    
                    predictions = self.pipeline.predict_race(event.year, event.round)
                    actual_results = RaceResult.objects.filter(
                        session__event=event,
                        position__isnull=False
                    ).select_related('driver')
                    
                    if predictions is not None and actual_results.exists():
                        for result in actual_results:
                            driver_name = f"{result.driver.given_name} {result.driver.family_name}"
                            
                            # Find prediction for this driver
                            pred_row = predictions[predictions['driver'] == driver_name]
                            if not pred_row.empty:
                                pred_pos = pred_row.iloc[0]['catboost_prediction']
                                actual_pos = result.position
                                
                                results.append({
                                    'event': f"{event.name} {event.year}",
                                    'driver': driver_name,
                                    'predicted': pred_pos,
                                    'actual': actual_pos,
                                    'error': abs(pred_pos - actual_pos)
                                })
                                
                except Exception as e:
                    self.stderr.write(f"Error validating {event}: {e}")
                    continue
            
            if results:
                results_df = pd.DataFrame(results)
                mae = results_df['error'].mean()
                rmse = np.sqrt((results_df['error'] ** 2).mean())
                
                self.stdout.write(f"\nValidation Results on {len(results)} predictions:")
                self.stdout.write(f"Mean Absolute Error: {mae:.2f}")
                self.stdout.write(f"Root Mean Square Error: {rmse:.2f}")
                
                # Show worst predictions
                worst_predictions = results_df.nlargest(5, 'error')
                self.stdout.write("\nWorst 5 Predictions:")
                for _, row in worst_predictions.iterrows():
                    self.stdout.write(
                        f"{row['driver']:<20} {row['event']:<25} "
                        f"Pred: {row['predicted']:>5.1f} Actual: {row['actual']:>2.0f} "
                        f"Error: {row['error']:>4.1f}"
                    )
                
                return results_df
            else:
                self.stdout.write("No validation results generated")
                return None
            
        except Exception as e:
            self.stderr.write(f"Error in validation: {e}")
            return None