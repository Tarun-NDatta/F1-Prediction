# prediction/management/commands/predict_xgboost.py
import joblib
import os
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from prediction.data_prep.pipeline import F1DataPipeline
from data.models import Event, Session, RaceResult, Driver, QualifyingResult, Team, DriverPerformance, TeamPerformance, xgboostprediction
import traceback
import glob

class Command(BaseCommand):
    help = 'Make XGBoost predictions with enhanced features and save to xgboostprediction table'

    # Define model directory and file patterns
    MODEL_DIR = r"C:\Users\tarun\diss\td188"
    MODEL_PREFIX = "ensemble_xgb_ridge"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = None
        self.preprocessor = None
        self.driver_features_list = [
            'reliability_score', 'rivalry_performance', 'teammate_battle',
            'qualifying_avg', 'points_per_race', 'quali_race_delta',
            'circuit_affinity', 'position_variance', 'position_momentum',
            'development_slope'
        ]
        self.team_features_list = [
            'reliability_score', 'development_slope', 'pit_stop_std',
            'moving_avg_5', 'position_variance', 'qualifying_avg'
        ]
    
    def add_arguments(self, parser):
        parser.add_argument('--year', type=int, required=True,
                            help='Year to predict')
        parser.add_argument('--round', type=int, required=True,
                            help='Round number to predict')
        parser.add_argument('--apply-adjustments', action='store_true',
                            help='Apply post-prediction race adjustments')
        parser.add_argument('--compare', action='store_true',
                            help='Compare predictions with actual results')
        parser.add_argument('--model-version', type=str, default='latest',
                            help='Specific model version to use (or "latest")')

    def find_model_files(self):
        """Find the model files based on the naming pattern"""
        model_path = os.path.join(self.MODEL_DIR, f"{self.MODEL_PREFIX}.pkl")
        features_path = os.path.join(self.MODEL_DIR, f"{self.MODEL_PREFIX}_features.pkl")
        preprocessor_path = os.path.join(self.MODEL_DIR, f"{self.MODEL_PREFIX}_preprocessing.pkl")
        
        if not all(os.path.exists(path) for path in [model_path, features_path, preprocessor_path]):
            raise FileNotFoundError("Required model files not found in expected location")
            
        return model_path, features_path, preprocessor_path

    def handle(self, *args, **options):
        try:
            # Find and load model files automatically
            model_path, features_path, preprocessor_path = self.find_model_files()
            
            self.stdout.write(f"Loading model from {model_path}...")
            model = joblib.load(model_path)
            feature_names = joblib.load(features_path)

            self.stdout.write("Loading preprocessor...")
            self.preprocessor = joblib.load(preprocessor_path)

            year = options['year']
            round_num = options['round']
            self.stdout.write(f"\nPreparing prediction data for {year} Round {round_num}...")

            event = Event.objects.get(year=year, round=round_num)
            self.stdout.write(f"Event: {event.name}")

            X_pred, drivers = self.prepare_prediction_data(feature_names, year, round_num)
            if X_pred is None or len(drivers) == 0:
                self.stdout.write(self.style.ERROR("No prediction data available"))
                return

            X_pred = self.preprocessor.transform(X_pred)
            self.stdout.write(f"Prediction samples: {len(drivers)}")
            self.stdout.write(f"Features: {len(feature_names)}")

            raw_predictions = model.predict(X_pred)
            position_predictions = self.convert_predictions_to_positions(raw_predictions, drivers)

            if options['apply_adjustments']:
                final_predictions = self.apply_race_adjustments(position_predictions, drivers, year, round_num)
            else:
                final_predictions = position_predictions

            results_df = pd.DataFrame({
                'driver': drivers,
                'predicted_position': final_predictions
            }).sort_values('predicted_position').reset_index(drop=True)
            
            self.display_predictions(results_df, event)

            if options['compare']:
                self.compare_with_actual(results_df, event)

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
            traceback.print_exc()

    def prepare_prediction_data(self, feature_names, year, round_num):
        try:
            qual_results = QualifyingResult.objects.filter(
                session__event__year=year,
                session__event__round=round_num
            ).select_related('driver', 'team').order_by('position')

            if not qual_results.exists():
                return None, []

            features = []
            drivers = []

            for result in qual_results:
                driver_features = self.get_driver_features(
                    result.driver,
                    result.team,
                    year,
                    round_num,
                    feature_names
                )
                if driver_features is not None:
                    features.append(driver_features)
                    drivers.append(f"{result.driver.given_name} {result.driver.family_name}")

            return np.array(features), drivers

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Data preparation error: {str(e)}"))
            return None, []

    def get_driver_features(self, driver, team, year, round_num, feature_names):
        try:
            features = {}

            driver_perf = DriverPerformance.objects.filter(
                driver=driver,
                event__year=year,
                event__round=round_num
            ).first()

            if driver_perf:
                for feature in self.driver_features_list:
                    if hasattr(driver_perf, feature):
                        features[feature] = getattr(driver_perf, feature)

            team_perf = TeamPerformance.objects.filter(
                team=team,
                event__year=year,
                event__round=round_num
            ).first()

            if team_perf:
                for feature in self.team_features_list:
                    if hasattr(team_perf, feature):
                        features[feature] = getattr(team_perf, feature)

            features.update(self.calculate_dynamic_features(driver, team, year, round_num))

            return [features.get(name, 0) for name in feature_names]

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting features for {driver}: {str(e)}"))
            return None

    def calculate_dynamic_features(self, driver, team, year, round_num):
        features = {}
        try:
            last_races = RaceResult.objects.filter(
                driver=driver,
                session__event__year__lte=year,
                session__event__round__lt=round_num
            ).order_by('-session__event__year', '-session__event__round')[:5]

            if last_races.exists():
                positions = [r.position for r in last_races if r.position is not None]
                if positions:
                    features['moving_avg_5'] = np.mean(positions[-5:]) if len(positions) >= 5 else np.mean(positions)
                    features['position_variance'] = np.var(positions)

            circuit = Event.objects.get(year=year, round=round_num).circuit
            circuit_results = RaceResult.objects.filter(
                driver=driver,
                session__event__circuit=circuit
            ).exclude(session__event__year=year, session__event__round=round_num)

            if circuit_results.exists():
                circuit_positions = [r.position for r in circuit_results if r.position is not None]
                if circuit_positions:
                    features['circuit_affinity'] = np.mean(circuit_positions)

            teammate_results = RaceResult.objects.filter(
                session__event__year=year,
                session__event__round=round_num,
                team=team
            ).exclude(driver=driver)

            if teammate_results.exists():
                teammate_pos = teammate_results.first().position
                driver_pos_obj = RaceResult.objects.filter(
                    driver=driver,
                    session__event__year=year,
                    session__event__round=round_num
                ).first()
                driver_pos = driver_pos_obj.position if driver_pos_obj else None

                if teammate_pos is not None and driver_pos is not None:
                    features['teammate_battle'] = teammate_pos - driver_pos

            return features
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error calculating dynamic features: {str(e)}"))
            return {}

    def convert_predictions_to_positions(self, raw_predictions, drivers):
        """Convert predictions where HIGHER raw values mean BETTER positions"""
        try:
            pred_df = pd.DataFrame({
                'driver': drivers,
                'raw_prediction': raw_predictions
            })
            
            # Sort DESCENDING (higher values = better positions)
            pred_df = pred_df.sort_values('raw_prediction', ascending=False)
            
            # Assign positions 1-20
            pred_df['position'] = range(1, len(pred_df)+1)
            
            return pred_df['position'].values
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Position conversion error: {str(e)}"))
            return raw_predictions

    def apply_race_adjustments(self, predictions, drivers, year, round_num):
        """Apply domain knowledge adjustments to predicted positions while maintaining unique positions"""
        adjusted = np.copy(predictions)
        try:
            team_limits = {
                'Red Bull': (1, 6), 'Mercedes': (2, 8), 'Ferrari': (2, 8),
                'McLaren': (3, 10), 'Aston Martin': (5, 12), 'Alpine': (8, 15),
                'Williams': (12, 18), 'AlphaTauri': (10, 18), 
                'Alfa Romeo': (12, 19), 'Haas': (14, 20)
            }

            recent_top5 = RaceResult.objects.filter(
                position__lte=5,
                session__event__year__gte=year-1
            ).values_list('driver__family_name', flat=True)

            circuit = Event.objects.get(year=year, round=round_num).circuit
            circuit_top3 = RaceResult.objects.filter(
                position__lte=3,
                session__event__circuit=circuit,
                session__event__year__gte=year-3
            ).values_list('driver__family_name', flat=True)

            # Apply adjustments but track what positions are taken
            adjustments_made = []
            
            for i, driver in enumerate(drivers):
                family_name = driver.split()[-1]
                current_pos = adjusted[i]
                original_pos = current_pos

                try:
                    driver_obj = Driver.objects.get(family_name=family_name)
                    team_name = driver_obj.teams.filter(year=year).first().team.name

                    if team_name in team_limits:
                        min_pos, max_pos = team_limits[team_name]
                        adjusted[i] = np.clip(current_pos, min_pos, max_pos)
                except Exception:
                    pass

                if family_name in recent_top5:
                    adjusted[i] = max(1, adjusted[i] - 1)
                if family_name in circuit_top3:
                    adjusted[i] = max(1, adjusted[i] - 1)

                # Track if we made an adjustment
                if adjusted[i] != original_pos:
                    adjustments_made.append((i, original_pos, adjusted[i], driver))

            # Now fix duplicate positions by reassigning them sequentially
            adjusted = self.resolve_duplicate_positions(adjusted, drivers)

            return adjusted
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Adjustment error: {str(e)}"))
            return predictions

    def resolve_duplicate_positions(self, positions, drivers):
        """Resolve duplicate positions by reassigning them to maintain unique sequential positions"""
        adjusted = np.copy(positions)
        
        # Find duplicates and resolve them
        unique_positions = []
        position_map = {}
        
        # Sort by current position to maintain relative order
        sorted_indices = np.argsort(adjusted)
        
        next_available_position = 1
        for idx in sorted_indices:
            current_pos = adjusted[idx]
            
            # If this position is already taken, assign the next available position
            while next_available_position in unique_positions:
                next_available_position += 1
            
            adjusted[idx] = next_available_position
            unique_positions.append(next_available_position)
            next_available_position += 1
        
        return adjusted

    def display_predictions(self, results_df, event):
        """Display and save predictions - matches baseline format"""
        self.stdout.write(f"\n=== Predicted Results for {event.name} ===")
        self.stdout.write(f"{'Rank':<4} {'Driver':<20} {'Position':<10}")
        self.stdout.write("-" * 40)

        for i, row in results_df.iterrows():
            # Display as integer position (matches baseline format)
            self.stdout.write(f"{i+1:<4} {row['driver']:<20} {int(row['predicted_position'])}")

        # Save predictions to database
        saved_count = 0
        for _, row in results_df.iterrows():
            given, *family = row['driver'].split()
            family_name = ' '.join(family) if family else given

            driver_obj = Driver.objects.filter(given_name=given, family_name=family_name).first()
            if not driver_obj:
                self.stdout.write(self.style.WARNING(f"Driver {row['driver']} not found in DB. Skipping save."))
                continue

            xgboostprediction.objects.update_or_create(
                event=event,
                driver=driver_obj,
                defaults={
                    'year': event.year,
                    'round_number': event.round,
                    'predicted_position': int(row['predicted_position']),
                    'model_name': 'xgboost_regression'
                }
            )
            saved_count += 1

        self.stdout.write(self.style.SUCCESS(f"Saved {saved_count} predictions to xgboostprediction table"))

    def compare_with_actual(self, results_df, event):
        """Compare predictions with actual results"""
        try:
            race_results = RaceResult.objects.filter(
                session__event=event,
                position__isnull=False
            ).select_related('driver').order_by('position')

            if not race_results.exists():
                self.stdout.write("\nNo race results available for comparison")
                return

            actual_df = pd.DataFrame([{
                'driver': f"{r.driver.given_name} {r.driver.family_name}",
                'actual_position': r.position
            } for r in race_results])

            comparison_df = results_df.merge(
                actual_df, on='driver', how='inner'
            ).sort_values('actual_position')

            if comparison_df.empty:
                self.stdout.write("\nNo matching drivers for comparison")
                return

            updated_count = 0
            for _, row in comparison_df.iterrows():
                given, *family = row['driver'].split()
                family_name = ' '.join(family) if family else given

                driver_obj = Driver.objects.filter(given_name=given, family_name=family_name).first()
                if not driver_obj:
                    continue

                xgboostprediction.objects.update_or_create(
                    event=event,
                    driver=driver_obj,
                    defaults={
                        'year': event.year,
                        'round_number': event.round,
                        'predicted_position': int(row['predicted_position']),
                        'actual_position': int(row['actual_position']),
                        'model_name': 'xgboost_regression'
                    }
                )
                updated_count += 1

            self.stdout.write(self.style.SUCCESS(f"Updated {updated_count} records with actual results"))

            # Calculate and display evaluation metrics
            y_true = comparison_df['actual_position']
            y_pred = comparison_df['predicted_position']

            self.stdout.write("\n=== Enhanced Evaluation ===")
            self.stdout.write(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
            self.stdout.write(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
            self.stdout.write(f"R²: {r2_score(y_true, y_pred):.2f}")
            self.stdout.write(f"Spearman Correlation: {spearmanr(y_true, y_pred)[0]:.2f}")

            for n in [3, 5, 10]:
                top_n_actual = set(y_true.nsmallest(n).index)
                top_n_pred = set(y_pred.nsmallest(n).index)
                accuracy = len(top_n_actual & top_n_pred) / n
                self.stdout.write(f"Top-{n} Accuracy: {accuracy:.1%}")

            self.stdout.write("\n=== Detailed Comparison ===")
            self.stdout.write(f"{'Driver':<20} {'Predicted':<10} {'Actual':<8} {'Diff':<6}")
            self.stdout.write("-" * 45)

            for _, row in comparison_df.iterrows():
                diff = row['predicted_position'] - row['actual_position']
                self.stdout.write(
                    f"{row['driver']:<20} {int(row['predicted_position']):<10} "
                    f"{int(row['actual_position']):<8} {diff:+.0f}"
                )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Comparison error: {str(e)}"))
            traceback.print_exc()