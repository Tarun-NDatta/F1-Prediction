# prediction/management/commands/predict_baseline.py
import joblib
import os
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from prediction.data_prep.pipeline import F1DataPipeline
from data.models import Event, Session, RaceResult, Driver, QualifyingResult, Team, DriverPerformance, TeamPerformance
import traceback

class Command(BaseCommand):
    help = 'Make predictions using trained model with enhanced features'
    
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
        parser.add_argument('--model', type=str, default='ridge_baseline.pkl',
                          help='Path to the trained model file')
        parser.add_argument('--year', type=int, required=True,
                          help='Year to predict')
        parser.add_argument('--round', type=int, required=True,
                          help='Round number to predict')
        parser.add_argument('--compare', action='store_true',
                          help='Compare predictions with actual results')
        parser.add_argument('--apply-adjustments', action='store_true',
                          help='Apply post-prediction race adjustments')

    def handle(self, *args, **options):
        try:
            # 1. Load model and features
            model_path = options['model']
            self.stdout.write(f"Loading model from {model_path}...")
            model = joblib.load(model_path)
            feature_names = joblib.load(model_path.replace('.pkl', '_features.pkl'))

            # 2. Handle preprocessor
            preprocessor_path = model_path.replace('.pkl', '_preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.stdout.write("Loading existing preprocessor...")
                self.preprocessor = joblib.load(preprocessor_path)
            else:
                self.stdout.write("Creating new preprocessor...")
                self.pipeline = F1DataPipeline(
                    test_size=0.2,
                    random_state=42,
                    impute_strategy='median'
                )
                X_train, _, _, _ = self.pipeline.prepare_training_data()
                self.preprocessor = self.pipeline.get_preprocessing_pipeline()
                self.preprocessor.fit(X_train)
                joblib.dump(self.preprocessor, preprocessor_path)
                self.stdout.write(f"Saved preprocessor to {preprocessor_path}")

            # 3. Prepare prediction data
            year = options['year']
            round_num = options['round']
            self.stdout.write(f"\nPreparing prediction data for {year} Round {round_num}...")
            
            event = Event.objects.get(year=year, round=round_num)
            self.stdout.write(f"Event: {event.name}")

            # 4. Get and transform features
            X_pred, drivers = self.prepare_prediction_data(feature_names, year, round_num)
            if X_pred is None or len(drivers) == 0:
                self.stdout.write(self.style.ERROR("No prediction data available"))
                return

            X_pred = self.preprocessor.transform(X_pred)
            self.stdout.write(f"Prediction samples: {len(drivers)}")
            self.stdout.write(f"Features: {len(feature_names)}")

            # 5. Make predictions
            raw_predictions = model.predict(X_pred)
            predictions = (self.apply_race_adjustments(raw_predictions, drivers, year, round_num) 
                          if options['apply_adjustments'] else raw_predictions)

            # 6. Display results
            results_df = self.prepare_results(drivers, predictions)
            self.display_predictions(results_df, event)
            
            if options['compare']:
                self.compare_with_actual(results_df, event)

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
            traceback.print_exc()

    def prepare_prediction_data(self, feature_names, year, round_num):
        """Prepare prediction data matching training features"""
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
        """Create feature vector for a single driver"""
        try:
            features = {}
            
            # Driver performance features
            driver_perf = DriverPerformance.objects.filter(
                driver=driver,
                event__year=year,
                event__round=round_num
            ).first()
            
            if driver_perf:
                for feature in self.driver_features_list:
                    if hasattr(driver_perf, feature):
                        features[feature] = getattr(driver_perf, feature)
            
            # Team performance features
            team_perf = TeamPerformance.objects.filter(
                team=team,
                event__year=year,
                event__round=round_num
            ).first()
            
            if team_perf:
                for feature in self.team_features_list:
                    if hasattr(team_perf, feature):
                        features[feature] = getattr(team_perf, feature)
            
            # Dynamic features
            features.update(self.calculate_dynamic_features(driver, team, year, round_num))
            
            # Align with expected feature order
            return [features.get(name, 0) for name in feature_names]
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting features for {driver}: {str(e)}"))
            return None

    def calculate_dynamic_features(self, driver, team, year, round_num):
        """Calculate features that aren't in the performance tables"""
        features = {}
        
        try:
            # Recent form (last 3 races)
            last_races = RaceResult.objects.filter(
                driver=driver,
                session__event__year__lte=year,
                session__event__round__lt=round_num
            ).order_by('-session__event__year', '-session__event__round')[:3]
            
            if last_races.exists():
                positions = [r.position for r in last_races if r.position is not None]
                if positions:
                    features['moving_avg_5'] = np.mean(positions[-5:]) if len(positions) >= 5 else np.mean(positions)
                    features['position_variance'] = np.var(positions)
            
            # Circuit affinity
            circuit = Event.objects.get(year=year, round=round_num).circuit
            circuit_results = RaceResult.objects.filter(
                driver=driver,
                session__event__circuit=circuit
            ).exclude(session__event__year=year, session__event__round=round_num)
            
            if circuit_results.exists():
                circuit_positions = [r.position for r in circuit_results if r.position is not None]
                if circuit_positions:
                    features['circuit_affinity'] = np.mean(circuit_positions)
            
            # Teammate comparison
            teammate_results = RaceResult.objects.filter(
                session__event__year=year,
                session__event__round=round_num,
                team=team
            ).exclude(driver=driver)
            
            if teammate_results.exists():
                teammate_pos = teammate_results.first().position
                driver_pos = RaceResult.objects.filter(
                    driver=driver,
                    session__event__year=year,
                    session__event__round=round_num
                ).first().position
                
                if teammate_pos and driver_pos:
                    features['teammate_battle'] = teammate_pos - driver_pos
            
            return features
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error calculating dynamic features: {str(e)}"))
            return {}

    def apply_race_adjustments(self, predictions, drivers, year, round_num):
        """Apply domain knowledge adjustments"""
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
            
            for i, driver in enumerate(drivers):
                family_name = driver.split()[-1]
                
                # Apply team limits
                try:
                    team = Driver.objects.get(family_name=family_name).teams.get(
                        year=year).team.name
                    if team in team_limits:
                        adjusted[i] = np.clip(adjusted[i], *team_limits[team])
                except:
                    pass
                
                # Apply performance boosts
                if family_name in recent_top5:
                    adjusted[i] *= 0.93  # 7% boost
                if family_name in circuit_top3:
                    adjusted[i] *= 0.95  # 5% boost
            
            return adjusted
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Adjustment error: {str(e)}"))
            return predictions

    def prepare_results(self, drivers, predictions):
        """Create sorted results DataFrame"""
        return pd.DataFrame({
            'driver': drivers,
            'predicted_position': predictions
        }).sort_values('predicted_position').reset_index(drop=True)

    def display_predictions(self, results_df, event):
        """Format and display predictions"""
        self.stdout.write(f"\n=== Predicted Results for {event.name} ===")
        self.stdout.write(f"{'Rank':<4} {'Driver':<20} {'Position':<10}")
        self.stdout.write("-" * 40)
        
        for i, row in results_df.iterrows():
            self.stdout.write(f"{i+1:<4} {row['driver']:<20} {row['predicted_position']:.2f}")

    def compare_with_actual(self, predictions_df, event):
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
            
            comparison_df = predictions_df.merge(
                actual_df, on='driver', how='inner'
            ).sort_values('actual_position')
            
            if comparison_df.empty:
                self.stdout.write("\nNo matching drivers for comparison")
                return
            
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
                    f"{row['driver']:<20} {row['predicted_position']:<10.2f} "
                    f"{row['actual_position']:<8} {diff:+.2f}"
                )
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Comparison error: {str(e)}"))
            traceback.print_exc()