import joblib
import os
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from django.db import models
from prediction.data_prep.pipeline import F1DataPipeline
from data.models import Event, Session, RaceResult, Driver, QualifyingResult, Team, DriverPerformance, TeamPerformance, ridgeregression, xgboostprediction
import traceback
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Make XGBoost predictions using trained ensemble models with Ridge regression features'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = None
        self.preprocessor = None
        self.imputer = None
        self.driver_features_list = ['reliability_score', 'rivalry_performance', 'teammate_battle', 'qualifying_avg', 'points_per_race', 'quali_race_delta', 'circuit_affinity', 'position_variance', 'position_momentum', 'development_slope']
        self.team_features_list = ['reliability_score', 'development_slope', 'pit_stop_std', 'moving_avg_5', 'position_variance', 'qualifying_avg']

    def add_arguments(self, parser):
        parser.add_argument('--model', type=str, default='ensemble_xgb_ridge.pkl', help='Path to the trained ensemble model file')
        parser.add_argument('--year', type=int, required=True, help='Year to predict')
        parser.add_argument('--round', type=int, required=True, help='Round number to predict')
        parser.add_argument('--compare', action='store_true', help='Compare predictions with actual results')
        parser.add_argument('--apply-adjustments', action='store_true', help='Apply post-prediction race adjustments')
        parser.add_argument('--use-ridge-features', action='store_true', help='Use Ridge predictions as additional features (for residual learning models)')

    def handle(self, *args, **options):
        try:
            # 1. Load model and features
            model_path = options['model']
            if not os.path.exists(model_path):
                self.stdout.write(self.style.ERROR(f"Model file not found: {model_path}"))
                return
                
            self.stdout.write(f"Loading ensemble model from {model_path}...")
            model = joblib.load(model_path)
            
            feature_names_path = model_path.replace('.pkl', '_features.pkl')
            if not os.path.exists(feature_names_path):
                self.stdout.write(self.style.ERROR(f"Feature names file not found: {feature_names_path}"))
                return
            feature_names = joblib.load(feature_names_path)
            logger.info(f"Loaded feature names: {feature_names}")

            # 2. Load or create preprocessor and imputer
            if not self.load_preprocessing_components(model_path):
                return

            # 3. Prepare prediction data
            year = options['year']
            round_num = options['round']
            self.stdout.write(f"\nPreparing prediction data for {year} Round {round_num}...")
            
            try:
                event = Event.objects.get(year=year, round=round_num)
                self.stdout.write(f"Event: {event.name}")
            except Event.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"Event not found for {year} Round {round_num}"))
                return

            # 4. Get and transform features (including Ridge predictions if needed)
            X_pred, drivers = self.prepare_prediction_data(feature_names, year, round_num, options['use_ridge_features'])
            if X_pred is None or len(drivers) == 0:
                self.stdout.write(self.style.ERROR("No prediction data available"))
                return

            logger.info(f"X_pred shape before preprocessing: {X_pred.shape}")
            # Apply preprocessing
            try:
                if self.preprocessor:
                    # Ensure preprocessor is fitted
                    if not hasattr(self.preprocessor, 'named_steps') or not self.preprocessor.named_steps['scaler'].get_params()['copy']:
                        self.stdout.write(self.style.WARNING("Preprocessor not fitted, attempting to re-fit..."))
                        self.preprocessor = self.fit_preprocessor(model_path)
                        if not self.preprocessor:
                            self.stdout.write(self.style.ERROR("Failed to fit preprocessor"))
                            return
                    X_pred = self.preprocessor.transform(X_pred)
                if self.imputer:
                    X_pred = self.imputer.transform(X_pred)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Preprocessing error: {str(e)}"))
                logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
                return
            
            self.stdout.write(f"Prediction samples: {len(drivers)}")
            self.stdout.write(f"Features: {len(feature_names)}")

            # 5. Make predictions based on model type
            raw_predictions = self.make_ensemble_predictions(model, X_pred, options['use_ridge_features'])
            
            # 6. Convert raw predictions to positions
            position_predictions = self.convert_predictions_to_positions(raw_predictions, drivers)
            
            # Apply adjustments to positions if requested
            if options['apply_adjustments']:
                final_predictions = self.apply_race_adjustments(position_predictions, drivers, year, round_num)
            else:
                final_predictions = position_predictions
            
            predictions = final_predictions

            # 7. Display results
            results_df = self.prepare_results(drivers, predictions)
            self.display_predictions(results_df, event)
            
            if options['compare']:
                self.compare_with_actual(results_df, event)

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
            logger.error(f"Error in handle: {str(e)}", exc_info=True)
            traceback.print_exc()

    def fit_preprocessor(self, model_path):
        """Fit a new preprocessor using training data"""
        try:
            self.stdout.write("Creating new preprocessor...")
            self.pipeline = F1DataPipeline(test_size=0.2, random_state=42, impute_strategy='median')
            X_train, _, _, _ = self.pipeline.prepare_training_data()
            if X_train is None or X_train.shape[0] == 0:
                self.stdout.write(self.style.ERROR("No training data available from F1DataPipeline"))
                return None
            preprocessor = self.pipeline.get_preprocessing_pipeline()
            preprocessor.fit(X_train)
            preprocessor_path = model_path.replace('.pkl', '_preprocessor.pkl')
            joblib.dump(preprocessor, preprocessor_path)
            self.stdout.write(f"Saved new preprocessor to {preprocessor_path}")
            return preprocessor
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to fit preprocessor: {str(e)}"))
            logger.error(f"Preprocessor fitting error: {str(e)}", exc_info=True)
            return None

    def load_preprocessing_components(self, model_path):
        """Load preprocessor and imputer if they exist, create and fit if they don't"""
        preprocessor_path = model_path.replace('.pkl', '_preprocessor.pkl')
        imputer_path = model_path.replace('.pkl', '_imputer.pkl')
        
        try:
            if os.path.exists(preprocessor_path):
                self.stdout.write("Loading existing preprocessor...")
                self.preprocessor = joblib.load(preprocessor_path)
                # Check if preprocessor is fitted
                try:
                    if not hasattr(self.preprocessor, 'named_steps') or not hasattr(self.preprocessor.named_steps['scaler'], 'mean_'):
                        self.stdout.write(self.style.WARNING("Loaded preprocessor is not fitted, re-fitting..."))
                        self.preprocessor = self.fit_preprocessor(model_path)
                        if not self.preprocessor:
                            return False
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Loaded preprocessor is not fitted: {str(e)}"))
                    self.preprocessor = self.fit_preprocessor(model_path)
                    if not self.preprocessor:
                        return False
            else:
                self.preprocessor = self.fit_preprocessor(model_path)
                if not self.preprocessor:
                    return False
            
            if os.path.exists(imputer_path):
                self.stdout.write("Loading existing imputer...")
                self.imputer = joblib.load(imputer_path)
            
            return True
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading preprocessing components: {str(e)}"))
            logger.error(f"Preprocessing components error: {str(e)}", exc_info=True)
            return False

    def prepare_prediction_data(self, feature_names, year, round_num, use_ridge_features=False):
        """Prepare prediction data matching training features, optionally including Ridge predictions"""
        try:
            qual_results = QualifyingResult.objects.filter(
                session__event__year=year, session__event__round=round_num
            ).select_related('driver', 'team').order_by('position')
            
            if not qual_results.exists():
                self.stdout.write(self.style.WARNING(
                    f"No qualifying results found for {year} Round {round_num}. Using historical data..."
                ))
                # Fallback to most recent qualifying results
                qual_results = QualifyingResult.objects.filter(
                    session__event__year__lt=year
                ).order_by('-session__event__year', '-session__event__round')[:20]
                
                if not qual_results.exists():
                    self.stdout.write(self.style.ERROR("No historical qualifying data available"))
                    return None, []
            
            # Get Ridge predictions for this event if using ridge features
            ridge_predictions = {}
            if use_ridge_features:
                ridge_results = ridgeregression.objects.filter(year=year, round_number=round_num).select_related('driver')
                ridge_predictions = {f"{r.driver.given_name} {r.driver.family_name}": r.predicted_position for r in ridge_results}
                
                if ridge_predictions:
                    self.stdout.write(f"Found {len(ridge_predictions)} Ridge predictions to use as features")
                else:
                    self.stdout.write("Warning: No Ridge predictions found, but use-ridge-features enabled")
            
            features = []
            drivers = []
            
            for result in qual_results:
                driver_name = f"{result.driver.given_name} {result.driver.family_name}"
                driver_features = self.get_driver_features(
                    result.driver, result.team, year, round_num, feature_names, 
                    ridge_predictions.get(driver_name) if use_ridge_features else None
                )
                
                if driver_features is not None:
                    features.append(driver_features)
                    drivers.append(driver_name)
            
            X_pred = np.array(features)
            logger.info(f"Prepared X_pred with shape: {X_pred.shape}, drivers: {len(drivers)}")
            return X_pred, drivers
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Data preparation error: {str(e)}"))
            logger.error(f"Data preparation error: {str(e)}", exc_info=True)
            return None, []

    def get_driver_features(self, driver, team, year, round_num, feature_names, ridge_prediction=None):
        """Create feature vector for a single driver, optionally including Ridge prediction"""
        try:
            features = {}
            
            # Driver performance features
            driver_perf = DriverPerformance.objects.filter(
                driver=driver, event__year=year, event__round=round_num
            ).first()
            
            if driver_perf:
                for feature in self.driver_features_list:
                    if hasattr(driver_perf, feature):
                        features[feature] = getattr(driver_perf, feature) or 0
            
            # Team performance features
            team_perf = TeamPerformance.objects.filter(
                team=team, event__year=year, event__round=round_num
            ).first()
            
            if team_perf:
                for feature in self.team_features_list:
                    if hasattr(team_perf, feature):
                        features[feature] = getattr(team_perf, feature) or 0
            
            # Dynamic features
            features.update(self.calculate_dynamic_features(driver, team, year, round_num))
            
            # Add Ridge prediction as feature if provided
            if ridge_prediction is not None:
                features['ridge_prediction'] = ridge_prediction
                
                # Calculate Ridge error from historical data
                historical_ridge = ridgeregression.objects.filter(
                    driver=driver, actual_position__isnull=False
                ).order_by('-year', '-round_number')[:5]
                
                if historical_ridge.exists():
                    errors = [abs(r.predicted_position - r.actual_position) for r in historical_ridge]
                    features['ridge_historical_mae'] = np.mean(errors)
                    features['ridge_historical_std'] = np.std(errors) if len(errors) > 1 else 0
                else:
                    features['ridge_historical_mae'] = 3.0
                    features['ridge_historical_std'] = 2.0
            
            # Align with expected feature order and fill missing features with 0
            aligned_features = []
            for name in feature_names:
                if name in features:
                    aligned_features.append(features[name])
                else:
                    self.stdout.write(self.style.WARNING(f"Feature {name} missing for {driver}, using 0"))
                    aligned_features.append(0)
            
            return aligned_features
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting features for {driver}: {str(e)}"))
            logger.error(f"Feature error for {driver}: {str(e)}", exc_info=True)
            return None

    def calculate_dynamic_features(self, driver, team, year, round_num):
        """Calculate features that aren't in the performance tables"""
        features = {}
        
        try:
            # Recent form (last 3 races)
            last_races = RaceResult.objects.filter(
                driver=driver, session__event__year__lte=year, session__event__round__lt=round_num
            ).order_by('-session__event__year', '-session__event__round')[:3]
            
            if last_races.exists():
                positions = [r.position for r in last_races if r.position is not None]
                if positions:
                    features['recent_avg_position'] = np.mean(positions)
                    features['position_trend'] = positions[0] - positions[-1] if len(positions) > 1 else 0
                    features['recent_best'] = min(positions)
                    features['recent_worst'] = max(positions)
            
            # Season performance
            season_results = RaceResult.objects.filter(
                driver=driver, session__event__year=year, session__event__round__lt=round_num, position__isnull=False
            )
            
            if season_results.exists():
                season_positions = [r.position for r in season_results]
                features['season_avg'] = np.mean(season_positions)
                features['season_best'] = min(season_positions)
                features['season_consistency'] = np.std(season_positions) if len(season_positions) > 1 else 0
            
            # Circuit affinity
            circuit = Event.objects.get(year=year, round=round_num).circuit
            circuit_results = RaceResult.objects.filter(
                driver=driver, session__event__circuit=circuit, position__isnull=False
            ).exclude(session__event__year=year, session__event__round=round_num)
            
            if circuit_results.exists():
                circuit_positions = [r.position for r in circuit_results]
                features['circuit_avg'] = np.mean(circuit_positions)
                features['circuit_best'] = min(circuit_positions)
                features['circuit_races'] = len(circuit_positions)
            else:
                features['circuit_avg'] = 10.0
                features['circuit_best'] = 10.0
                features['circuit_races'] = 0
            
            # Qualifying vs Race performance
            quali_results = QualifyingResult.objects.filter(
                driver=driver, session__event__year=year, position__isnull=False
            )[:5]
            
            if quali_results.exists():
                quali_positions = [q.position for q in quali_results]
                features['quali_avg'] = np.mean(quali_positions)
                features['quali_best'] = min(quali_positions)
            
            return features
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error calculating dynamic features: {str(e)}"))
            logger.error(f"Dynamic features error: {str(e)}", exc_info=True)
            return {}

    def make_ensemble_predictions(self, model, X_pred, use_ridge_features=False):
        """Make predictions with ensemble model, handling different ensemble types"""
        try:
            if hasattr(model, 'get_ridge_model') and hasattr(model, 'get_xgb_model'):
                self.stdout.write("Using residual learning ensemble...")
                return model.predict(X_pred)
            
            elif hasattr(model, 'predict'):
                self.stdout.write("Using standard ensemble model...")
                return model.predict(X_pred)
            
            else:
                raise ValueError("Unknown model type - cannot make predictions")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Prediction error: {str(e)}"))
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return model.predict(X_pred)

    def convert_predictions_to_positions(self, raw_predictions, drivers):
        """Convert raw model predictions to race positions (1-20)"""
        try:
            pred_df = pd.DataFrame({'driver': drivers, 'raw_prediction': raw_predictions})
            pred_df = pred_df.sort_values('raw_prediction').reset_index(drop=True)
            pred_df['position'] = range(1, len(pred_df) + 1)
            return pred_df['position'].values
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Position conversion error: {str(e)}"))
            logger.error(f"Position conversion error: {str(e)}", exc_info=True)
            return raw_predictions

    def apply_race_adjustments(self, predictions, drivers, year, round_num):
        """Apply domain knowledge adjustments to POSITIONS"""
        adjusted = np.copy(predictions)
        
        try:
            team_limits = {
                'Red Bull Racing': (1, 4), 'Mercedes': (2, 8), 'Ferrari': (2, 8),
                'McLaren': (3, 10), 'Aston Martin': (5, 12), 'Alpine': (8, 15),
                'Williams': (12, 18), 'AlphaTauri': (10, 18), 'Alfa Romeo': (12, 19),
                'Haas': (14, 20)
            }
            
            ridge_reliability = {}
            ridge_results = ridgeregression.objects.filter(year=year, round_number=round_num).select_related('driver')
            for rr in ridge_results:
                driver_name = f"{rr.driver.given_name} {rr.driver.family_name}"
                ridge_reliability[driver_name] = rr.predicted_position
            
            circuit = Event.objects.get(year=year, round=round_num).circuit
            circuit_specialists = self.get_circuit_specialists(circuit, year)
            
            for i, driver in enumerate(drivers):
                current_pos = adjusted[i]
                
                try:
                    driver_obj = Driver.objects.get(given_name=driver.split()[0], family_name=driver.split()[-1])
                    recent_team = driver_obj.teams.filter(year=year).first()
                    
                    if recent_team and recent_team.team.name in team_limits:
                        min_pos, max_pos = team_limits[recent_team.team.name]
                        adjusted[i] = np.clip(current_pos, min_pos, max_pos)
                except:
                    pass
                
                if driver in circuit_specialists:
                    adjusted[i] = max(1, adjusted[i] - 1)
                
                if driver in ridge_reliability:
                    ridge_pos = ridge_reliability[driver]
                    if abs(current_pos - ridge_pos) > 5:
                        adjusted[i] = (current_pos + ridge_pos) / 2
            
            return adjusted
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Adjustment error: {str(e)}"))
            logger.error(f"Adjustment error: {str(e)}", exc_info=True)
            return predictions

    def get_circuit_specialists(self, circuit, current_year):
        """Identify drivers who perform well at this circuit"""
        specialists = []
        
        try:
            circuit_results = RaceResult.objects.filter(
                session__event__circuit=circuit,
                session__event__year__gte=current_year-3,
                position__lte=3
            ).select_related('driver').values('driver__given_name', 'driver__family_name').annotate(
                avg_position=models.Avg('position'),
                race_count=models.Count('id')
            ).filter(race_count__gte=2)
            
            specialists = [f"{r['driver__given_name']} {r['driver__family_name']}" for r in circuit_results]
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Circuit specialist detection error: {str(e)}"))
            logger.error(f"Circuit specialist error: {str(e)}", exc_info=True)
        
        return specialists

    def prepare_results(self, drivers, predictions):
        """Create sorted results DataFrame"""
        return pd.DataFrame({'driver': drivers, 'predicted_position': predictions}).sort_values('predicted_position').reset_index(drop=True)

    def display_predictions(self, results_df, event):
        """Format and display predictions with integer positions"""
        self.stdout.write(f"\n=== XGBoost Ensemble Predicted Results for {event.name} ===")
        self.stdout.write(f"{'Rank':<4} {'Driver':<20} {'Position':<10}")
        self.stdout.write("-" * 40)
        
        for i, row in results_df.iterrows():
            self.stdout.write(f"{i+1:<4} {row['driver']:<20} {int(row['predicted_position'])}")

        saved_count = 0
        for _, row in results_df.iterrows():
            try:
                driver_parts = row['driver'].split()
                given = driver_parts[0]
                family = ' '.join(driver_parts[1:]) if len(driver_parts) > 1 else driver_parts[0]
                driver_obj = Driver.objects.filter(given_name=given, family_name=family).first()
                    
                if driver_obj:
                    xgboostprediction.objects.update_or_create(
                        event=event, driver=driver_obj,
                        defaults={'predicted_position': int(row['predicted_position'])}
                    )
                    saved_count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error saving prediction for {row['driver']}: {str(e)}"))
                logger.error(f"Prediction save error for {row['driver']}: {str(e)}", exc_info=True)

        self.stdout.write(self.style.SUCCESS(f"Saved {saved_count} predictions to XGBoostPrediction table"))

    def compare_with_actual(self, predictions_df, event):
        """Compare predictions with actual results"""
        try:
            race_results = RaceResult.objects.filter(
                session__event=event, position__isnull=False
            ).select_related('driver').order_by('position')
            
            if not race_results.exists():
                self.stdout.write("\nNo race results available for comparison")
                return
            
            actual_df = pd.DataFrame([
                {'driver': f"{r.driver.given_name} {r.driver.family_name}", 'actual_position': r.position}
                for r in race_results
            ])
            
            comparison_df = predictions_df.merge(actual_df, on='driver', how='inner').sort_values('actual_position')
            
            if comparison_df.empty:
                self.stdout.write("\nNo matching drivers for comparison")
                return
            
            y_true = comparison_df['actual_position']
            y_pred = comparison_df['predicted_position']
            
            self.stdout.write("\n=== XGBoost Model Evaluation ===")
            self.stdout.write(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
            self.stdout.write(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
            self.stdout.write(f"R²: {r2_score(y_true, y_pred):.3f}")
            
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            self.stdout.write(f"Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
            
            for n in [3, 5, 10]:
                top_n_actual = set(y_true.nsmallest(n).index)
                top_n_pred = set(y_pred.nsmallest(n).index)
                accuracy = len(top_n_actual & top_n_pred) / n
                self.stdout.write(f"Top-{n} Accuracy: {accuracy:.1%}")
            
            for tolerance in [2, 3, 5]:
                within_tolerance = np.mean(np.abs(y_true - y_pred) <= tolerance)
                self.stdout.write(f"Within ±{tolerance} positions: {within_tolerance:.1%}")
            
            ridge_results = ridgeregression.objects.filter(
                event=event, actual_position__isnull=False
            ).select_related('driver')
            
            if ridge_results.exists():
                self.stdout.write("\n=== Comparison with Ridge Baseline ===")
                ridge_df = pd.DataFrame([
                    {'driver': f"{r.driver.given_name} {r.driver.family_name}", 
                     'ridge_prediction': r.predicted_position, 
                     'actual_position': r.actual_position}
                    for r in ridge_results
                ])
                
                full_comparison = comparison_df.merge(ridge_df[['driver', 'ridge_prediction']], on='driver', how='inner')
                
                if not full_comparison.empty:
                    ridge_mae = mean_absolute_error(full_comparison['actual_position'], full_comparison['ridge_prediction'])
                    xgb_mae = mean_absolute_error(full_comparison['actual_position'], full_comparison['predicted_position'])
                    
                    improvement = ridge_mae - xgb_mae
                    improvement_pct = (improvement / ridge_mae) * 100 if ridge_mae > 0 else 0
                    
                    self.stdout.write(f"Ridge MAE: {ridge_mae:.2f}")
                    self.stdout.write(f"XGBoost MAE: {xgb_mae:.2f}")
                    self.stdout.write(f"Improvement: {improvement:.2f} ({improvement_pct:+.1f}%)")
            
            self.stdout.write("\n=== Detailed Comparison ===")
            self.stdout.write(f"{'Driver':<20} {'Predicted':<10} {'Actual':<8} {'Diff':<6}")
            self.stdout.write("-" * 45)
            
            for _, row in comparison_df.iterrows():
                diff = row['predicted_position'] - row['actual_position']
                self.stdout.write(f"{row['driver']:<20} {int(row['predicted_position']):<10} {int(row['actual_position']):<8} {diff:+.0f}")
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Comparison error: {str(e)}"))
            logger.error(f"Comparison error: {str(e)}", exc_info=True)
            traceback.print_exc()