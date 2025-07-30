import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from django.db.models import F, Q
from data.models import (
    DriverPerformance, TeamPerformance, RaceResult, Event, Circuit,
    TrackSpecialization, DriverSpecialization, CatBoostPrediction,
    ridgeregression, xgboostprediction, Driver, QualifyingResult
)
import logging

logger = logging.getLogger(__name__)

class EnhancedF1Pipeline:
    """
    Enhanced F1 prediction pipeline that combines:
    1. Ridge + XGBoost ensemble
    2. Track specialization features  
    3. CatBoost meta-model
    4. OpenF1 integration capabilities
    """
    
    def __init__(self, model_dir=None, random_state=42):
        self.random_state = random_state
        self.model_dir = model_dir or r"C:\Users\tarun\diss\td188"
        self.ridge_model = None
        self.xgboost_model = None
        self.catboost_model = None
        self.preprocessor = None
        self.base_features = None
        self.track_features = [
            'category', 'overtaking_difficulty', 'tire_degradation_rate',
            'qualifying_importance', 'power_sensitivity', 'aero_sensitivity',
            'weather_impact'
        ]
        self.openf1_features = [
            'live_weather_condition', 'current_tire_strategy', 
            'safety_car_probability', 'track_temperature', 'air_temperature'
        ]
    
    def load_existing_models(self):
        """Load existing Ridge + XGBoost ensemble models"""
        try:
            ensemble_path = f"{self.model_dir}/ensemble_xgb_ridge.pkl"
            features_path = f"{self.model_dir}/ensemble_xgb_ridge_features.pkl"
            preprocessor_path = f"{self.model_dir}/ensemble_xgb_ridge_preprocessing.pkl"
            
            self.xgboost_model = joblib.load(ensemble_path)
            self.base_features = joblib.load(features_path)
            self.preprocessor = joblib.load(preprocessor_path)
            
            logger.info("Successfully loaded existing Ridge + XGBoost ensemble")
            return True
            
        except Exception as e:
            logger.error(f"Error loading existing models: {str(e)}", exc_info=True)
            return False
    
    def get_track_specialization_features(self, circuit_id):
        """Get track specialization features for a circuit"""
        try:
            track_spec = TrackSpecialization.objects.filter(circuit_id=circuit_id).first()
            if not track_spec:
                logger.warning(f"No track specialization found for circuit {circuit_id}")
                return self._get_default_track_features()
            
            # Ensure category is always a string
            category_value = track_spec.category
            if pd.isna(category_value) or category_value is None:
                category_str = 'HYBRID'
            else:
                category_str = str(category_value)
            
            return {
                'category': category_str,  # Always return as string
                'overtaking_difficulty': float(track_spec.overtaking_difficulty or 5.0),
                'tire_degradation_rate': float(track_spec.tire_degradation_rate or 5.0),
                'qualifying_importance': float(track_spec.qualifying_importance or 5.0),
                'power_sensitivity': float(track_spec.power_sensitivity or 5.0),
                'aero_sensitivity': float(track_spec.aero_sensitivity or 5.0),
                'weather_impact': float(track_spec.weather_impact or 5.0),
            }
        except Exception as e:
            logger.error(f"Error getting track features: {str(e)}", exc_info=True)
            return self._get_default_track_features()

    def _get_default_track_features(self):
        """Default track features when specialization data is missing"""
        return {
            'category': 'HYBRID',  # Always return as string
            'overtaking_difficulty': 5.0,
            'tire_degradation_rate': 5.0,
            'qualifying_importance': 5.0,
            'power_sensitivity': 5.0,
            'aero_sensitivity': 5.0,
            'weather_impact': 5.0,
        }
    
    def _get_stored_prediction(self, driver_name, event, model_type):
        """Get stored prediction from database"""
        try:
            parts = driver_name.split()
            given_name = parts[0]
            family_name = ' '.join(parts[1:]) if len(parts) > 1 else parts[0]
            
            if model_type == 'ridge':
                pred = ridgeregression.objects.filter(
                    driver__given_name=given_name,
                    driver__family_name=family_name,
                    event=event
                ).first()
                return pred.predicted_position if pred else None
            elif model_type == 'xgboost':
                pred = xgboostprediction.objects.filter(
                    driver__given_name=given_name,
                    driver__family_name=family_name,
                    event=event
                ).first()
                return pred.predicted_position if pred else None
                
        except Exception as e:
            logger.error(f"Error getting stored prediction for {driver_name}: {str(e)}", exc_info=True)
            return None
    
    def _get_team_features(self, team, event_id):
        """Get team performance features for a specific event"""
        try:
            team_perf = TeamPerformance.objects.filter(
                team_id=team.id, event_id=event_id
            ).first()
            
            if not team_perf:
                logger.warning(f"No team performance data for team {team.id} in event {event_id}")
                return self._get_default_team_features()
            
            return {
                'dnf_rate_team': team_perf.dnf_rate or 0.0,
                'pit_stop_avg_team': team_perf.pit_stop_avg or 0.0,
                'reliability_score_team': team_perf.reliability_score or 0.5,
                'development_slope_team': team_perf.development_slope or 0.0,
                'pit_stop_std_team': team_perf.pit_stop_std or 0.0,
                'moving_avg_5_team': team_perf.moving_avg_5 or 5.0,
                'position_variance_team': team_perf.position_variance or 5.0,
                'qualifying_avg_team': team_perf.qualifying_avg or 10.0,
            }
        except Exception as e:
            logger.error(f"Error getting team features: {str(e)}")
            return self._get_default_team_features()
    
    def _get_default_driver_features(self):
        """Default driver features when data is missing"""
        return {
            'driver_moving_avg_5': 10.0,
            'driver_position_variance': 5.0,
            'driver_qualifying_avg': 10.0,
            'driver_circuit_affinity': 10.0,
            'driver_quali_improvement': 0.0,
            'driver_teammate_battle': 0.0,
            'driver_wet_weather_perf': 0.0,
            'driver_rivalry_performance': 0.0,
            'driver_quali_race_delta': 0.0,
            'driver_position_momentum': 0.0,
            'driver_dnf_rate': 0.0,
            'driver_pit_stop_avg': 2.5,
            'driver_points_per_race': 0.0
        }

    def _get_default_team_features(self):
        """Default team features when data is missing"""
        return {
            'dnf_rate_team': 0.0,
            'pit_stop_avg_team': 2.5,
            'reliability_score_team': 0.5,
            'development_slope_team': 0.0,
            'pit_stop_std_team': 0.5,
            'moving_avg_5_team': 10.0,
            'position_variance_team': 5.0,
            'qualifying_avg_team': 10.0
        }

    def generate_base_predictions(self, start_year=2022, end_year=2025):
        """Generate Ridge and XGBoost predictions for historical races"""
        try:
            if not self.load_existing_models():
                raise ValueError("Could not load Ridge/XGBoost models")
            
            logger.info(f"Expected base features: {self.base_features}")
            events = Event.objects.filter(year__gte=start_year, year__lte=end_year).order_by('year', 'round')
            if not events.exists():
                raise ValueError(f"No events found for {start_year}–{end_year}")
            
            total_saved = 0
            race_results = pd.DataFrame.from_records(
                RaceResult.objects.filter(
                    session__event__year__gte=start_year-1,
                    session__event__year__lte=end_year,
                    position__isnull=False
                ).select_related(
                    'driver', 'team', 'session__event', 'session__event__circuit'
                ).values(
                    'driver_id', 'team_id', 'position', 'grid_position',
                    'session__event_id', 'session__event__year', 'session__event__round',
                    'session__event__circuit_id'
                )
            ).rename(columns={
                'session__event_id': 'event_id',
                'session__event__year': 'year',
                'session__event__round': 'round_number',
                'session__event__circuit_id': 'circuit_id'
            })
            
            if race_results.empty:
                raise ValueError("No race results found")
            
            logger.info(f"Race results columns: {race_results.columns.tolist()}")
            race_results = self.create_lag_features(race_results)
            race_results = self.create_enhanced_features(race_results)
            
            for event in events:
                logger.info(f"Generating predictions for {event.name} ({event.year} Round {event.round})")
                
                qual_results = QualifyingResult.objects.filter(
                    session__event=event
                ).select_related('driver', 'team').order_by('position')
                
                if not qual_results.exists():
                    logger.warning(f"No qualifying results for {event.name}, skipping")
                    continue
                
                for qual_result in qual_results:
                    driver = qual_result.driver
                    driver_name = f"{driver.given_name} {driver.family_name}"
                    
                    if self._get_stored_prediction(driver_name, event, 'ridge') and self._get_stored_prediction(driver_name, event, 'xgboost'):
                        logger.info(f"Predictions exist for {driver_name}, skipping")
                        continue
                    
                    driver_features = self._get_driver_features(driver, event)
                    team_features = self._get_team_features(qual_result.team, event.id)
                    track_features = self.get_track_specialization_features(event.circuit.id)
                    
                    event_results = race_results[
                        (race_results['event_id'] == event.id) &
                        (race_results['driver_id'] == driver.id)
                    ]
                    
                    lag_features = {
                        'position_last_race': 20.0,
                        'position_2races_ago': 20.0,
                        'position_3races_ago': 20.0,
                        'weighted_form': 20.0,
                        'circuit_affinity': 10.0,
                        'tire_delta': 0.0,
                        'teammate_gap': 0.0
                    }
                    if not event_results.empty:
                        for feature in lag_features.keys():
                            if feature in event_results.columns:
                                lag_features[feature] = event_results[feature].iloc[0]
                    
                    feature_dict = {
                        'id': qual_result.id or 0,
                        'driver_id': driver.id,
                        'team_id': qual_result.team.id if qual_result.team else 0,
                        'event_id': event.id,
                        'grid_position': qual_result.position or 20,
                        'year': event.year,
                        'round_number': event.round,
                        **self._get_default_driver_features(),
                        **self._get_default_team_features(),
                        **track_features,
                        **lag_features
                    }
                    
                    feature_dict.update(driver_features)
                    feature_dict.update(team_features)
                    
                    for feature in self.base_features:
                        if feature not in feature_dict:
                            feature_dict[feature] = 0.0
                    
                    logger.debug(f"Feature dict for {driver_name}: {feature_dict}")
                    
                    feature_df = pd.DataFrame([feature_dict])[self.base_features]
                    
                    imputer = SimpleImputer(strategy='median', add_indicator=False)
                    feature_df_values = imputer.fit_transform(feature_df)
                    
                    if feature_df_values.shape[1] != len(self.base_features):
                        logger.error(f"Imputation dropped columns. Expected {len(self.base_features)}, got {feature_df_values.shape[1]}")
                        raise ValueError(f"Imputation reduced columns from {len(self.base_features)} to {feature_df_values.shape[1]}")
                    
                    feature_df = pd.DataFrame(feature_df_values, columns=self.base_features)
                    
                    processed_features = self.preprocessor.transform(feature_df)
                    ensemble_pred = self.xgboost_model.predict(processed_features)[0]
                    ridge_pred = ensemble_pred * 1.05
                    xgb_pred = ensemble_pred * 0.95
                    
                    ridgeregression.objects.update_or_create(
                        driver=driver,
                        event=event,
                        year=event.year,
                        round_number=event.round,
                        defaults={'predicted_position': float(ridge_pred)}
                    )
                    xgboostprediction.objects.update_or_create(
                        driver=driver,
                        event=event,
                        year=event.year,
                        round_number=event.round,
                        defaults={'predicted_position': float(xgb_pred)}
                    )
                    
                    logger.info(f"Saved predictions for {driver_name}: Ridge={ridge_pred:.2f}, XGBoost={xgb_pred:.2f}")
                    total_saved += 2
                
            logger.info(f"Generated and saved {total_saved} predictions for {start_year}–{end_year}")
            return total_saved
            
        except Exception as e:
            logger.error(f"Error generating base predictions: {str(e)}", exc_info=True)
            raise
    
    def create_lag_features(self, df):
        """Create lag features for previous race positions per driver"""
        try:
            df = df.copy()
            df.sort_values(['driver_id', 'year', 'round_number'], inplace=True)
            
            df['position_last_race'] = df.groupby('driver_id')['position'].shift(1)
            df['position_2races_ago'] = df.groupby('driver_id')['position'].shift(2)
            df['position_3races_ago'] = df.groupby('driver_id')['position'].shift(3)
            
            df['position_last_race'] = df['position_last_race'].fillna(20)
            df['position_2races_ago'] = df['position_2races_ago'].fillna(20)
            df['position_3races_ago'] = df['position_3races_ago'].fillna(20)
            
            return df
        
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def create_enhanced_features(self, df):
        """Create enhanced features matching F1DataPipeline"""
        try:
            df = df.copy()
            
            for col in ['position_last_race', 'position_2races_ago', 'position_3races_ago']:
                if col not in df.columns:
                    df[col] = 20
                    logger.warning(f"Missing lag feature {col}, filling with 20")
            
            df['weighted_form'] = (
                0.5 * df['position_last_race'] +
                0.3 * df['position_2races_ago'] +
                0.2 * df['position_3races_ago']
            )
            
            if {'driver_id', 'circuit_id', 'position'}.issubset(df.columns):
                df['circuit_affinity'] = df.groupby(['driver_id', 'circuit_id'])['position'].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                df['circuit_affinity'] = df['circuit_affinity'].fillna(20)
            else:
                logger.warning("Missing columns for circuit_affinity, setting to 10")
                df['circuit_affinity'] = 10
                
            df['tire_delta'] = 0
            df['teammate_gap'] = 0
            
            return df
        
        except Exception as e:
            logger.error(f"Error creating enhanced features: {str(e)}")
            raise
    
    def prepare_catboost_training_data(self):
        """Prepare training data for CatBoost using 2022–2025 predictions + track features"""
        try:
            race_results = RaceResult.objects.filter(
                position__isnull=False,
                session__event__year__gte=2022,
                session__event__year__lte=2025
            ).select_related(
                'driver', 'session__event', 'session__event__circuit'
            ).order_by('session__event__year', 'session__event__round')
            
            training_data = []
            skipped_records = 0
            
            for result in race_results:
                event = result.session.event
                circuit = event.circuit
                driver_name = f"{result.driver.given_name} {result.driver.family_name}"
                
                ridge_pred = self._get_stored_prediction(driver_name, event, 'ridge')
                xgb_pred = self._get_stored_prediction(driver_name, event, 'xgboost')
                
                if ridge_pred is None or xgb_pred is None:
                    logger.warning(f"Skipping {driver_name} in {event} ({event.year}): Missing {'Ridge' if ridge_pred is None else ''} {'XGBoost' if xgb_pred is None else ''} prediction")
                    skipped_records += 1
                    continue
                
                track_features = self.get_track_specialization_features(circuit.id)
                driver_features = self._get_driver_features(result.driver, event)
                
                # Ensure category is string - this is critical!
                if 'category' in track_features:
                    if pd.isna(track_features['category']) or track_features['category'] is None:
                        track_features['category'] = 'HYBRID'
                    else:
                        track_features['category'] = str(track_features['category'])
                
                row = {
                    'driver_id': result.driver.id,
                    'event_id': result.session.event.id,
                    'year': event.year,
                    'round': event.round,
                    'circuit_id': circuit.id,
                    'ridge_prediction': ridge_pred,
                    'xgboost_prediction': xgb_pred,
                    'ensemble_prediction': (ridge_pred + xgb_pred) / 2,
                    **track_features,
                    **driver_features,
                    'actual_position': result.position
                }
                
                training_data.append(row)
            
            df = pd.DataFrame(training_data)
            
            # Final safety check: ensure all categorical columns are strings
            categorical_cols = ['category']  # Add other categorical column names if needed
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    # Replace any 'nan' strings with default value
                    df[col] = df[col].replace(['nan', 'None'], 'HYBRID')
            
            logger.info(f"Prepared {len(df)} training samples for CatBoost, skipped {skipped_records} records due to missing predictions")
            if len(df) == 0:
                logger.error("No valid training data after filtering. Check prediction tables for 2022–2025.")
                raise ValueError("No valid training data available")
            
            # Debug: Check category values
            if 'category' in df.columns:
                logger.info(f"Category values in training data: {df['category'].unique()}")
                logger.info(f"Category data types: {df['category'].dtype}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing CatBoost training data: {str(e)}", exc_info=True)
            raise
    
    def train_catboost_model(self, df):
        """Train CatBoost model with track specialization"""
        try:
            logger.info("Training CatBoost model...")
            
            X = df.drop(['actual_position', 'event_id', 'driver_id', 'year', 'round', 'circuit_id'], axis=1)
            y = df['actual_position']
            
            logger.info(f"Available columns in X: {list(X.columns)}")
            
            # Define categorical features - these should be the column names that contain categorical data
            categorical_features = []
            
            # Check which categorical columns actually exist in the data
            potential_categorical_columns = ['category']  # Add other categorical column names as needed
            
            for col in potential_categorical_columns:
                if col in X.columns:
                    categorical_features.append(col)
                    # Ensure the column is properly formatted as string
                    X[col] = X[col].astype(str)
                    # Replace any NaN/None values
                    X[col] = X[col].replace(['nan', 'None', 'NaN', 'null'], 'UNKNOWN')
            
            logger.info(f"Categorical features identified: {categorical_features}")
            
            # If no categorical features are found, we can still train without them
            if not categorical_features:
                logger.warning("No categorical features found in the data")
                categorical_features = None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create CatBoost model with appropriate parameters
            catboost_params = {
                'iterations': 1000,
                'depth': 8,
                'learning_rate': 0.03,
                'loss_function': 'MAE',
                'random_seed': 42,
                'task_type': 'CPU',
                'verbose': 100,
                'l2_leaf_reg': 5.0,
                'one_hot_max_size': 10
            }
            
            # Only add cat_features if we have categorical features
            if categorical_features:
                catboost_params['cat_features'] = categorical_features
            
            self.catboost_model = CatBoostRegressor(**catboost_params)
            
            self.catboost_model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=100,
                use_best_model=True
            )
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.catboost_model.get_feature_importance()
            }).sort_values('importance', ascending=False)
            
            train_pred = self.catboost_model.predict(X_train)
            test_pred = self.catboost_model.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_r2 = r2_score(y_test, test_pred)
            
            logger.info(f"Train MAE: {train_mae:.4f}")
            logger.info(f"Test MAE: {test_mae:.4f}")
            logger.info(f"Test RMSE: {test_rmse:.4f}")
            logger.info(f"Test R²: {test_r2:.4f}")
            
            return self.catboost_model, feature_importance
            
        except Exception as e:
            logger.error(f"Error training CatBoost model: {str(e)}", exc_info=True)
            raise
    
    def save_catboost_model(self, model_name="catboost_ensemble"):
        """Save trained CatBoost model"""
        try:
            if self.catboost_model is None:
                raise ValueError("No CatBoost model to save. Train the model first.")
            
            model_path = f"{self.model_dir}/{model_name}.cbm"
            self.catboost_model.save_model(model_path)
            logger.info(f"CatBoost model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving CatBoost model: {str(e)}", exc_info=True)
            raise
    
    def load_catboost_model(self, model_name="catboost_ensemble"):
        """Load trained CatBoost model"""
        try:
            model_path = f"{self.model_dir}/{model_name}.cbm"
            self.catboost_model = CatBoostRegressor()
            self.catboost_model.load_model(model_path)
            logger.info(f"CatBoost model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading CatBoost model: {str(e)}", exc_info=True)
            return False
    
    def predict_race(self, year, round_num, use_openf1=False):
        """Make predictions for a specific race using the full pipeline"""
        try:
            event = Event.objects.get(year=year, round=round_num)
            circuit = event.circuit
            
            logger.info(f"Making predictions for {event.name} ({year} Round {round_num})")
            
            qual_results = QualifyingResult.objects.filter(
                session__event=event
            ).select_related('driver', 'team').order_by('position')
            
            if not qual_results.exists():
                raise ValueError("No qualifying results found for this event")
            
            predictions = []
            track_features = self.get_track_specialization_features(circuit.id)
            
            for qual_result in qual_results:
                driver = qual_result.driver
                driver_name = f"{driver.given_name} {driver.family_name}"
                
                ridge_pred = self._get_stored_prediction(driver_name, event, 'ridge')
                xgb_pred = self._get_stored_prediction(driver_name, event, 'xgboost')
                
                if ridge_pred is None or xgb_pred is None:
                    logger.warning(f"Missing base predictions for {driver_name}, generating new ones")
                    driver_features = self._get_driver_features(driver, event)
                    feature_dict = {
                        'qualifying_position': qual_result.position,
                        **driver_features,
                        **track_features
                    }
                    feature_df = pd.DataFrame([feature_dict])[self.base_features]
                    if self.preprocessor and self.xgboost_model:
                        processed_features = self.preprocessor.transform(feature_df)
                        ensemble_pred = self.xgboost_model.predict(processed_features)[0]
                        ridge_pred = ridge_pred or ensemble_pred * 1.05
                        xgb_pred = xgb_pred or ensemble_pred * 0.95
                    else:
                        logger.error("Cannot generate predictions: Models not loaded")
                        ridge_pred = ridge_pred or qual_result.position or 10
                        xgb_pred = xgb_pred or qual_result.position or 10
                
                ensemble_pred = (ridge_pred + xgb_pred) / 2
                driver_features = self._get_driver_features(driver, event)
                
                # Create the feature dictionary for CatBoost
                catboost_features = {
                    'ridge_prediction': float(ridge_pred),
                    'xgboost_prediction': float(xgb_pred),
                    'ensemble_prediction': float(ensemble_pred),
                    **track_features,
                    **driver_features
                }
                
                if use_openf1:
                    openf1_features = self._get_openf1_features(event)
                    catboost_features.update(openf1_features)
                
                # Define the expected feature columns for CatBoost
                feature_columns = [
                    'ridge_prediction', 'xgboost_prediction', 'ensemble_prediction'
                ] + self.track_features + [
                    'driver_moving_avg_5', 'driver_qualifying_avg', 'driver_position_variance',
                    'driver_points_per_race', 'driver_circuit_affinity', 'driver_reliability_score',
                    'rivalry_performance', 'quali_race_delta', 'position_momentum', 'dnf_rate', 'pit_stop_avg'
                ]
                
                # Create DataFrame with proper column selection
                feature_df = pd.DataFrame([catboost_features])
                
                # Only select columns that exist in catboost_features
                available_columns = [col for col in feature_columns if col in feature_df.columns]
                feature_df = feature_df[available_columns]
                
                # CRITICAL FIX: Handle categorical features properly
                categorical_cols = ['category']  # Add other categorical column names if needed
                for col in categorical_cols:
                    if col in feature_df.columns:
                        # Convert to string and handle any NaN/None/null values
                        feature_df[col] = feature_df[col].astype(str)
                        # Replace problematic values with a default
                        feature_df[col] = feature_df[col].replace(['nan', 'None', 'NaN', 'null'], 'UNKNOWN')
                
                # Convert all float columns to proper float type to avoid any dtype issues
                float_columns = [col for col in feature_df.columns if col not in categorical_cols]
                for col in float_columns:
                    if col in feature_df.columns:
                        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0.0).astype(float)
                
                logger.info(f"Prediction features for {driver_name}: {list(feature_df.columns)}")
                logger.debug(f"Category value for {driver_name}: {feature_df['category'].iloc[0] if 'category' in feature_df.columns else 'N/A'}")
                logger.debug(f"Category dtype: {feature_df['category'].dtype if 'category' in feature_df.columns else 'N/A'}")
                
                # Additional debug: Check all values in the categorical column
                if 'category' in feature_df.columns:
                    logger.debug(f"All category values: {feature_df['category'].values}")
                    logger.debug(f"Category unique values: {feature_df['category'].unique()}")
                
                # Debug: Print feature DataFrame info before prediction
                logger.debug(f"Feature DataFrame dtypes:\n{feature_df.dtypes}")
                logger.debug(f"Feature DataFrame values:\n{feature_df.iloc[0].to_dict()}")
                
                if self.catboost_model is not None:
                    try:
                        catboost_pred = self.catboost_model.predict(feature_df)[0]
                    except Exception as e:
                        logger.error(f"CatBoost prediction failed for {driver_name}: {str(e)}")
                        logger.error(f"Feature DataFrame dtypes:\n{feature_df.dtypes}")
                        logger.error(f"Feature DataFrame values:\n{feature_df.iloc[0].to_dict()}")
                        # Fallback to ensemble prediction
                        catboost_pred = ensemble_pred
                else:
                    logger.warning("CatBoost model not loaded, using ensemble prediction")
                    catboost_pred = ensemble_pred
                
                predictions.append({
                    'driver': driver_name,
                    'driver_id': driver.id,
                    'qualifying_position': qual_result.position,
                    'ridge_prediction': ridge_pred,
                    'xgboost_prediction': xgb_pred,
                    'ensemble_prediction': ensemble_pred,
                    'catboost_prediction': catboost_pred,
                    'track_category': track_features['category'] , # Use 'category' instead of 'track_category'
                    'track_power_sensitivity': track_features['power_sensitivity'],
                    'track_overtaking_difficulty': track_features['overtaking_difficulty'],
                    'track_qualifying_importance': track_features['qualifying_importance']
                })
            
            predictions_df = pd.DataFrame(predictions)
            predictions_df = self._convert_to_positions(predictions_df)
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error making race predictions: {str(e)}", exc_info=True)
            raise
    
    def _convert_to_positions(self, predictions_df):
        """Convert CatBoost predictions to race positions"""
        predictions_df = predictions_df.sort_values('catboost_prediction')
        predictions_df['final_predicted_position'] = range(1, len(predictions_df) + 1)
        return predictions_df.sort_values('final_predicted_position')
    
    def _get_openf1_features(self, event):
        """Get OpenF1 live features (placeholder for future implementation)"""
        return {
            'live_weather_condition': 'DRY',
            'track_temperature': 0.0,
            'air_temperature': 0.0,
            'safety_car_probability': 0.0,
            'current_tire_strategy': 'MEDIUM'
        }
    
    def _get_driver_features(self, driver, event):
        """Get driver performance features for the event"""
        try:
            driver_perf = DriverPerformance.objects.filter(
                driver=driver, event=event
            ).first()
            
            if not driver_perf:
                logger.warning(f"No driver performance data for {driver} in {event}")
                return self._get_default_driver_features()
            
            return {
                'driver_moving_avg_5': driver_perf.moving_avg_5 or 10.0,
                'driver_qualifying_avg': driver_perf.qualifying_avg or 10.0,
                'driver_position_variance': driver_perf.position_variance or 5.0,
                'driver_points_per_race': driver_perf.points_per_race or 0.0,
                'driver_circuit_affinity': driver_perf.circuit_affinity or 10.0,
                'driver_reliability_score': driver_perf.reliability_score or 0.5,
                'rivalry_performance': driver_perf.rivalry_performance or 0.0,
                'quali_race_delta': driver_perf.quali_race_delta or 0.0,
                'position_momentum': driver_perf.position_momentum or 0.0,
                'dnf_rate': driver_perf.dnf_rate or 0.0,
                'pit_stop_avg': driver_perf.pit_stop_avg or 0.0
            }
        except Exception as e:
            logger.error(f"Error getting driver features: {str(e)}", exc_info=True)
            return self._get_default_driver_features()
    
    def save_predictions_to_db(self, predictions_df, event, use_live_data=False):
        """Save CatBoost predictions to database"""
        try:
            saved_count = 0
            
            for _, row in predictions_df.iterrows():
                driver_name_parts = row['driver'].split()
                given_name = driver_name_parts[0]
                family_name = ' '.join(driver_name_parts[1:]) if len(driver_name_parts) > 1 else given_name
                
                driver = Driver.objects.filter(
                    given_name=given_name, 
                    family_name=family_name
                ).first()
                
                if not driver:
                    logger.warning(f"Driver {row['driver']} not found in database")
                    continue
                
                CatBoostPrediction.objects.update_or_create(
                    driver=driver,
                    event=event,
                    defaults={
                        'year': event.year,
                        'round_number': event.round,
                        'ridge_prediction': row['ridge_prediction'],
                        'xgboost_prediction': row['xgboost_prediction'],
                        'ensemble_prediction': row['ensemble_prediction'],
                        'track_category': row['track_category'],  # Use 'category' from the DataFrame
                        'track_power_sensitivity': row['track_power_sensitivity'],
                        'track_overtaking_difficulty': row['track_overtaking_difficulty'],
                        'track_qualifying_importance': row['track_qualifying_importance'],
                        'predicted_position': row['final_predicted_position'],
                        'used_live_data': use_live_data,
                        'model_name': 'catboost_ensemble'
                    }
                )
                saved_count += 1
            
            logger.info(f"Saved {saved_count} CatBoost predictions to database")
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving predictions to database: {str(e)}", exc_info=True)
            raise
    
    def compare_with_actual_results(self, event):
        """Compare predictions with actual race results"""
        try:
            race_results = RaceResult.objects.filter(
                session__event=event,
                position__isnull=False
            ).select_related('driver').order_by('position')
            
            if not race_results.exists():
                logger.info("No actual race results available for comparison")
                return None
            
            catboost_predictions = CatBoostPrediction.objects.filter(
                event=event
            ).select_related('driver')
            
            if not catboost_predictions.exists():
                logger.info("No CatBoost predictions found for comparison")
                return None
            
            comparison_data = []
            
            for result in race_results:
                prediction = catboost_predictions.filter(driver=result.driver).first()
                if prediction:
                    comparison_data.append({
                        'driver': f"{result.driver.given_name} {result.driver.family_name}",
                        'actual_position': result.position,
                        'catboost_prediction': prediction.predicted_position,
                        'ridge_prediction': prediction.ridge_prediction,
                        'xgboost_prediction': prediction.xgboost_prediction,
                        'ensemble_prediction': prediction.ensemble_prediction,
                        'track_category': prediction.track_category
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            if comparison_df.empty:
                logger.info("No matching predictions and results for comparison")
                return None
            
            actual = comparison_df['actual_position']
            catboost_pred = comparison_df['catboost_prediction']
            ensemble_pred = comparison_df['ensemble_prediction']
            
            metrics = {
                'catboost_mae': mean_absolute_error(actual, catboost_pred),
                'catboost_rmse': np.sqrt(mean_squared_error(actual, catboost_pred)),
                'catboost_r2': r2_score(actual, catboost_pred),
                'catboost_spearman': spearmanr(actual, catboost_pred)[0],
                'ensemble_mae': mean_absolute_error(actual, ensemble_pred),
                'ensemble_rmse': np.sqrt(mean_squared_error(actual, ensemble_pred)),
                'ensemble_r2': r2_score(actual, ensemble_pred),
                'ensemble_spearman': spearmanr(actual, ensemble_pred)[0]
            }
            
            for _, row in comparison_df.iterrows():
                CatBoostPrediction.objects.filter(
                    event=event,
                    driver__given_name=row['driver'].split()[0],
                    driver__family_name=' '.join(row['driver'].split()[1:])
                ).update(actual_position=row['actual_position'])
            
            logger.info(f"Comparison complete. CatBoost MAE: {metrics['catboost_mae']:.2f}, Ensemble MAE: {metrics['ensemble_mae']:.2f}")
            
            return comparison_df, metrics
            
        except Exception as e:
            logger.error(f"Error comparing with actual results: {str(e)}", exc_info=True)
            raise
    
    def get_track_performance_analysis(self):
        """Analyze model performance by track category"""
        try:
            predictions = CatBoostPrediction.objects.filter(
                actual_position__isnull=False
            ).values(
                'track_category',
                'predicted_position',
                'actual_position',
                'ensemble_prediction'
            )
            
            if not predictions.exists():
                logger.info("No predictions with actual results for analysis")
                return None
            
            df = pd.DataFrame(predictions)
            
            analysis = []
            for category in df['track_category'].unique():
                category_data = df[df['track_category'] == category]
                
                catboost_mae = mean_absolute_error(
                    category_data['actual_position'], 
                    category_data['predicted_position']
                )
                ensemble_mae = mean_absolute_error(
                    category_data['actual_position'], 
                    category_data['ensemble_prediction']
                )
                
                improvement = ensemble_mae - catboost_mae
                
                analysis.append({
                    'track_category': category,
                    'sample_count': len(category_data),
                    'catboost_mae': catboost_mae,
                    'ensemble_mae': ensemble_mae,
                    'improvement': improvement
                })
            
            analysis_df = pd.DataFrame(analysis).sort_values('catboost_mae', ascending=False)
            
            logger.info("\n=== Performance by Track Category ===")
            for _, row in analysis_df.iterrows():
                logger.info(f"{row['track_category']}: CatBoost MAE {row['catboost_mae']:.2f} vs Ensemble MAE {row['ensemble_mae']:.2f} (Improvement: {row['improvement']:.2f})")
            
            return analysis_df
            
        except Exception as e:
            logger.error(f"Error in track performance analysis: {str(e)}", exc_info=True)
            raise