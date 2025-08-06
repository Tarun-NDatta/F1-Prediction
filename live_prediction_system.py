"""
Live Prediction System for F1 Races
Integrates with OpenF1 API and continuously updates predictions until 15 laps to go
"""

import os
import sys
import django
import asyncio
import aiohttp
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Dissertation.settings')
django.setup()

from django.db import transaction
from django.utils import timezone
from data.models import (
    Event, Driver, Team, Circuit, Session, RaceResult, QualifyingResult,
    DriverPerformance, TeamPerformance, TrackCharacteristics,
    ridgeregression, xgboostprediction, CatBoostPrediction,
    TrackSpecialization, DriverSpecialization
)
from prediction.data_prep.utilities import load_model
from prediction.data_prep.pipeline import FeaturePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OpenF1Client:
    """Client for OpenF1 API to get live race data via GitHub"""
    
    def __init__(self):
        # Get GitHub credentials from environment variables with mock defaults
        self.github_username = os.getenv('GITHUB_USERNAME', 'mock_username')
        self.github_token = os.getenv('GITHUB_TOKEN', 'mock_token')
        self.github_repo = os.getenv('GITHUB_REPO', 'mock_repo')
        
        # GitHub API base URL
        self.github_api_url = "https://api.github.com"
        self.session = None
        
        # Mock data for testing (remove when real credentials are added)
        self.use_mock_data = (self.github_username == 'mock_username' or 
                             self.github_token == 'mock_token' or 
                             self.github_repo == 'mock_repo')
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_current_session(self) -> Optional[Dict]:
        """Get current F1 session from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock session data
                return {
                    'session_id': 12345,
                    'session_name': 'Dutch Grand Prix Race',
                    'session_status': 'active',
                    'session_type': 'Race',
                    'date': '2025-08-24',
                    'total_laps': 50
                }
            
            # GitHub API call to get OpenF1 data
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # This would be the actual GitHub API call to your private repo
            # For now, returning mock data
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/sessions.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Parse the content from GitHub
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    sessions = json.loads(content)
                    
                    # Find current/active session
                    for session in sessions:
                        if session.get('session_status') == 'active':
                            return session
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting current session: {e}")
            return None
    
    async def get_lap_times(self, session_id: int, lap_number: int = None) -> List[Dict]:
        """Get lap times for a session from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock lap times data
                mock_lap_times = []
                for driver_num in range(1, 21):  # 20 drivers
                    mock_lap_times.append({
                        'driver_number': driver_num,
                        'lap_number': lap_number or 5,
                        'lap_duration': 85.0 + (driver_num * 0.5),  # Mock lap times
                        'sector1_time': 28.0 + (driver_num * 0.1),
                        'sector2_time': 29.0 + (driver_num * 0.1),
                        'sector3_time': 28.0 + (driver_num * 0.3),
                    })
                return mock_lap_times
            
            # GitHub API call for lap times
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/lap_times.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    lap_times = json.loads(content)
                    
                    # Filter by session_id and lap_number if provided
                    filtered_times = [lt for lt in lap_times if lt.get('session_id') == session_id]
                    if lap_number:
                        filtered_times = [lt for lt in filtered_times if lt.get('lap_number') == lap_number]
                    
                    return filtered_times
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return []
            return []
        except Exception as e:
            logger.error(f"Error getting lap times: {e}")
            return []
    
    async def get_driver_positions(self, session_id: int) -> List[Dict]:
        """Get current driver positions from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock driver positions
                mock_positions = []
                for driver_num in range(1, 21):  # 20 drivers
                    mock_positions.append({
                        'driver_number': driver_num,
                        'position': driver_num,  # Mock positions
                        'last_lap_time': 85.0 + (driver_num * 0.5),
                        'sector_times': [
                            28.0 + (driver_num * 0.1),
                            29.0 + (driver_num * 0.1),
                            28.0 + (driver_num * 0.3)
                        ]
                    })
                return mock_positions
            
            # GitHub API call for driver positions
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/positions.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    positions = json.loads(content)
                    
                    # Filter by session_id
                    filtered_positions = [pos for pos in positions if pos.get('session_id') == session_id]
                    return filtered_positions
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return []
            return []
        except Exception as e:
            logger.error(f"Error getting driver positions: {e}")
            return []
    
    async def get_weather_data(self, session_id: int) -> Optional[Dict]:
        """Get weather data for session from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock weather data
                return {
                    'air_temp': 22.5,
                    'track_temp': 28.0,
                    'humidity': 65.0,
                    'rain': False,
                    'wind_speed': 8.5,
                    'wind_direction': 180
                }
            
            # GitHub API call for weather data
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/weather.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    weather_data = json.loads(content)
                    
                    # Filter by session_id
                    filtered_weather = [w for w in weather_data if w.get('session_id') == session_id]
                    return filtered_weather[0] if filtered_weather else None
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return None
    
    async def get_tire_data(self, session_id: int) -> List[Dict]:
        """Get tire compound data from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock tire data
                mock_tire_data = []
                compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
                for driver_num in range(1, 21):
                    mock_tire_data.append({
                        'driver_number': driver_num,
                        'compound': compounds[driver_num % len(compounds)],
                        'age': driver_num % 20,  # Mock tire age
                        'wear': (driver_num * 5) % 100  # Mock tire wear
                    })
                return mock_tire_data
            
            # GitHub API call for tire data
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/tyres.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    tire_data = json.loads(content)
                    
                    # Filter by session_id
                    filtered_tires = [t for t in tire_data if t.get('session_id') == session_id]
                    return filtered_tires
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return []
            return []
        except Exception as e:
            logger.error(f"Error getting tire data: {e}")
            return []

class LivePredictionSystem:
    """Main live prediction system"""
    
    def __init__(self):
        self.openf1_client = None
        self.models = {}
        self.feature_pipeline = None
        self.current_event = None
        self.current_session = None
        self.prediction_interval = 30  # seconds
        self.final_prediction_lap = 15  # Make final prediction with 15 laps to go
        self.should_stop = False  # Flag to stop gracefully
        
        # Load ML models
        self._load_models()
        self._load_feature_pipeline()
    
    def _load_models(self):
        """Load all trained ML models"""
        try:
            models_dir = "models"
            
            # Load latest models (assuming most recent timestamp)
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if not model_files:
                logger.error("No model files found")
                return
            
            # Get latest timestamp
            timestamps = set()
            for file in model_files:
                if '_v' in file and '_' in file.split('_v')[1]:
                    timestamp = file.split('_v')[1].split('_')[0]
                    timestamps.add(timestamp)
            
            if not timestamps:
                logger.error("No valid model timestamps found")
                return
            
            latest_timestamp = max(timestamps)
            logger.info(f"Loading models with timestamp: {latest_timestamp}")
            
            # Load Ridge Regression
            ridge_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_ridge.pkl")
            if os.path.exists(ridge_path):
                self.models['ridge'] = load_model(ridge_path)
                logger.info("Loaded Ridge Regression model")
            
            # Load XGBoost
            xgb_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_xgboost.pkl")
            if os.path.exists(xgb_path):
                self.models['xgboost'] = load_model(xgb_path)
                logger.info("Loaded XGBoost model")
            
            # Load Stacked Model (CatBoost)
            stacked_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_stacked_model.pkl")
            if os.path.exists(stacked_path):
                self.models['catboost'] = load_model(stacked_path)
                logger.info("Loaded CatBoost ensemble model")
            
            # Load preprocessor
            preprocessor_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_preprocessor.pkl")
            if os.path.exists(preprocessor_path):
                self.models['preprocessor'] = load_model(preprocessor_path)
                logger.info("Loaded preprocessor")
            
            # Load feature list
            features_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_features.pkl")
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.models['feature_list'] = pickle.load(f)
                logger.info("Loaded feature list")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _load_feature_pipeline(self):
        """Load feature pipeline for live data processing"""
        try:
            self.feature_pipeline = FeaturePipeline()
            logger.info("Loaded feature pipeline")
        except Exception as e:
            logger.error(f"Error loading feature pipeline: {e}")
    
    async def get_current_race_event(self) -> Optional[Event]:
        """Get current race event from database"""
        try:
            # Get current date
            now = timezone.now().date()
            
            # Find current/upcoming race
            current_event = Event.objects.filter(
                date__gte=now,
                year=2025  # Adjust year as needed
            ).order_by('date').first()
            
            if current_event:
                logger.info(f"Current event: {current_event.name} ({current_event.date})")
                return current_event
            
            return None
        except Exception as e:
            logger.error(f"Error getting current race event: {e}")
            return None
    
    def _extract_live_features(self, live_data: Dict, event: Event) -> pd.DataFrame:
        """Extract features from live race data"""
        try:
            features = {}
            
            # Basic event features
            features['year'] = event.year
            features['round'] = event.round
            features['circuit_id'] = event.circuit.circuit_ref
            
            # Weather features
            if 'weather_data' in live_data and live_data['weather_data']:
                weather = live_data['weather_data']
                features['air_temp'] = weather.get('air_temp', 20.0)
                features['track_temp'] = weather.get('track_temp', 25.0)
                features['humidity'] = weather.get('humidity', 50.0)
                features['rain'] = 1 if weather.get('rain', False) else 0
            else:
                features['air_temp'] = 20.0
                features['track_temp'] = 25.0
                features['humidity'] = 50.0
                features['rain'] = 0
            
            # Lap and race progress
            features['current_lap'] = live_data.get('current_lap', 1)
            features['total_laps'] = live_data.get('total_laps', 50)
            features['race_progress'] = features['current_lap'] / features['total_laps']
            
            # Driver-specific features from live data
            driver_features = []
            for driver_data in live_data.get('driver_positions', []):
                driver_id = driver_data.get('driver_number')
                if not driver_id:
                    continue
                
                try:
                    driver = Driver.objects.get(driver_id=str(driver_id))
                    
                    # Get historical performance data
                    driver_perf = DriverPerformance.objects.filter(
                        driver=driver,
                        event__year__lt=event.year
                    ).order_by('-event__date').first()
                    
                    team_perf = TeamPerformance.objects.filter(
                        team=driver.team,
                        event__year__lt=event.year
                    ).order_by('-event__date').first()
                    
                    # Current race position
                    current_position = driver_data.get('position', 20)
                    
                    # Lap times (if available)
                    lap_time = driver_data.get('last_lap_time', 0)
                    sector_times = driver_data.get('sector_times', [0, 0, 0])
                    
                    driver_feature = {
                        'driver_id': driver.driver_id,
                        'driver_ref': driver.driver_ref,
                        'team_ref': driver.team.team_ref if driver.team else 'unknown',
                        'current_position': current_position,
                        'last_lap_time': lap_time,
                        'sector1_time': sector_times[0] if len(sector_times) > 0 else 0,
                        'sector2_time': sector_times[1] if len(sector_times) > 1 else 0,
                        'sector3_time': sector_times[2] if len(sector_times) > 2 else 0,
                        'moving_avg_5': driver_perf.moving_avg_5 if driver_perf else 10.0,
                        'qualifying_avg': driver_perf.qualifying_avg if driver_perf else 10.0,
                        'position_variance': driver_perf.position_variance if driver_perf else 5.0,
                        'points_per_race': driver_perf.points_per_race if driver_perf else 5.0,
                        'circuit_affinity': driver_perf.circuit_affinity if driver_perf else 10.0,
                        'reliability_score': driver_perf.reliability_score if driver_perf else 0.8,
                        'team_dnf_rate': team_perf.dnf_rate if team_perf else 0.1,
                        'team_pit_stop_avg': team_perf.pit_stop_avg if team_perf else 2.5,
                    }
                    
                    driver_features.append(driver_feature)
                    
                except Driver.DoesNotExist:
                    logger.warning(f"Driver not found: {driver_id}")
                    continue
            
            # Create DataFrame with all driver features
            if driver_features:
                df = pd.DataFrame(driver_features)
                
                # Add global features to each row
                for key, value in features.items():
                    df[key] = value
                
                return df
            else:
                logger.warning("No driver features extracted")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting live features: {e}")
            return pd.DataFrame()
    
    def _make_predictions(self, features_df: pd.DataFrame) -> Dict:
        """Make predictions using all loaded models"""
        try:
            if features_df.empty:
                logger.warning("No features available for prediction")
                return {}
            
            predictions = {}
            
            # Preprocess features
            if 'preprocessor' in self.models:
                try:
                    processed_features = self.models['preprocessor'].transform(features_df)
                except Exception as e:
                    logger.error(f"Error preprocessing features: {e}")
                    return {}
            else:
                processed_features = features_df
            
            # Ridge Regression predictions
            if 'ridge' in self.models:
                try:
                    ridge_preds = self.models['ridge'].predict(processed_features)
                    predictions['ridge'] = ridge_preds
                    logger.info("Made Ridge Regression predictions")
                except Exception as e:
                    logger.error(f"Error in Ridge Regression: {e}")
            
            # XGBoost predictions
            if 'xgboost' in self.models:
                try:
                    xgb_preds = self.models['xgboost'].predict(processed_features)
                    predictions['xgboost'] = xgb_preds
                    logger.info("Made XGBoost predictions")
                except Exception as e:
                    logger.error(f"Error in XGBoost: {e}")
            
            # CatBoost ensemble predictions
            if 'catboost' in self.models:
                try:
                    # Prepare ensemble features
                    ensemble_features = []
                    if 'ridge' in predictions:
                        ensemble_features.append(predictions['ridge'])
                    if 'xgboost' in predictions:
                        ensemble_features.append(predictions['xgboost'])
                    
                    if ensemble_features:
                        ensemble_input = np.column_stack(ensemble_features)
                        catboost_preds = self.models['catboost'].predict(ensemble_input)
                        predictions['catboost'] = catboost_preds
                        logger.info("Made CatBoost ensemble predictions")
                except Exception as e:
                    logger.error(f"Error in CatBoost: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {}
    
    def _save_predictions_to_db(self, predictions: Dict, event: Event, session: Session, 
                               live_data: Dict, is_final: bool = False):
        """Save predictions to database"""
        try:
            with transaction.atomic():
                current_lap = live_data.get('current_lap', 1)
                
                # Get driver mapping
                driver_mapping = {}
                for driver_data in live_data.get('driver_positions', []):
                    driver_id = driver_data.get('driver_number')
                    if driver_id:
                        try:
                            driver = Driver.objects.get(driver_id=str(driver_id))
                            driver_mapping[driver_id] = driver
                        except Driver.DoesNotExist:
                            continue
                
                # Save Ridge predictions
                if 'ridge' in predictions:
                    for i, pred in enumerate(predictions['ridge']):
                        if i < len(driver_mapping):
                            driver_id = list(driver_mapping.keys())[i]
                            driver = driver_mapping[driver_id]
                            
                            ridgeregression.objects.update_or_create(
                                driver=driver,
                                event=event,
                                model_name='ridge_regression_live',
                                defaults={
                                    'year': event.year,
                                    'round_number': event.round,
                                    'predicted_position': float(pred),
                                    'created_at': timezone.now()
                                }
                            )
                
                # Save XGBoost predictions
                if 'xgboost' in predictions:
                    for i, pred in enumerate(predictions['xgboost']):
                        if i < len(driver_mapping):
                            driver_id = list(driver_mapping.keys())[i]
                            driver = driver_mapping[driver_id]
                            
                            xgboostprediction.objects.update_or_create(
                                driver=driver,
                                event=event,
                                defaults={
                                    'year': event.year,
                                    'round_number': event.round,
                                    'predicted_position': float(pred),
                                    'created_at': timezone.now()
                                }
                            )
                
                # Save CatBoost predictions
                if 'catboost' in predictions:
                    for i, pred in enumerate(predictions['catboost']):
                        if i < len(driver_mapping):
                            driver_id = list(driver_mapping.keys())[i]
                            driver = driver_mapping[driver_id]
                            
                            # Get track specialization data
                            track_spec = TrackSpecialization.objects.filter(
                                circuit=event.circuit
                            ).first()
                            
                            CatBoostPrediction.objects.update_or_create(
                                driver=driver,
                                event=event,
                                model_name='catboost_ensemble_live',
                                defaults={
                                    'year': event.year,
                                    'round_number': event.round,
                                    'ridge_prediction': predictions.get('ridge', [0])[i] if 'ridge' in predictions else None,
                                    'xgboost_prediction': predictions.get('xgboost', [0])[i] if 'xgboost' in predictions else None,
                                    'predicted_position': float(pred),
                                    'prediction_confidence': 0.85,  # Default confidence
                                    'used_live_data': True,
                                    'weather_condition': 'DRY' if not live_data.get('weather_data', {}).get('rain') else 'WET',
                                    'tire_strategy_available': bool(live_data.get('tire_data')),
                                    'track_category': track_spec.category if track_spec else 'HYBRID',
                                    'track_power_sensitivity': track_spec.power_sensitivity if track_spec else 5.0,
                                    'track_overtaking_difficulty': track_spec.overtaking_difficulty if track_spec else 5.0,
                                    'track_qualifying_importance': track_spec.qualifying_importance if track_spec else 5.0,
                                    'created_at': timezone.now()
                                }
                            )
                
                logger.info(f"Saved {'final' if is_final else 'live'} predictions to database (Lap {current_lap})")
                
        except Exception as e:
            logger.error(f"Error saving predictions to database: {e}")
    
    async def _collect_live_data(self, session_id: int) -> Dict:
        """Collect all live data from OpenF1"""
        try:
            live_data = {}
            
            # Get current positions
            positions = await self.openf1_client.get_driver_positions(session_id)
            live_data['driver_positions'] = positions
            
            # Get weather data
            weather = await self.openf1_client.get_weather_data(session_id)
            live_data['weather_data'] = weather
            
            # Get tire data
            tire_data = await self.openf1_client.get_tire_data(session_id)
            live_data['tire_data'] = tire_data
            
            # Get latest lap times
            lap_times = await self.openf1_client.get_lap_times(session_id)
            if lap_times:
                # Find current lap number
                current_lap = max([lt.get('lap_number', 0) for lt in lap_times])
                live_data['current_lap'] = current_lap
                live_data['total_laps'] = 50  # Default, should be extracted from session data
                
                # Add lap times to driver positions
                for position in live_data['driver_positions']:
                    driver_number = position.get('driver_number')
                    if driver_number:
                        driver_laps = [lt for lt in lap_times if lt.get('driver_number') == driver_number]
                        if driver_laps:
                            latest_lap = max(driver_laps, key=lambda x: x.get('lap_number', 0))
                            position['last_lap_time'] = latest_lap.get('lap_duration', 0)
                            position['sector_times'] = [
                                latest_lap.get('sector1_time', 0),
                                latest_lap.get('sector2_time', 0),
                                latest_lap.get('sector3_time', 0)
                            ]
            
            return live_data
            
        except Exception as e:
            logger.error(f"Error collecting live data: {e}")
            return {}
    
    async def run_live_prediction(self):
        """Main method to run live prediction system"""
        try:
            logger.info("Starting Live Prediction System")
            
            # Get current race event
            self.current_event = await self.get_current_race_event()
            if not self.current_event:
                logger.warning("No current race event found")
                return
            
            # Initialize OpenF1 client
            async with OpenF1Client() as client:
                self.openf1_client = client
                
                # Get current session
                session_data = await self.openf1_client.get_current_session()
                if not session_data:
                    logger.warning("No active session found")
                    return
                
                session_id = session_data.get('session_id')
                logger.info(f"Active session found: {session_id}")
                
                # Main prediction loop
                while not self.should_stop:
                    try:
                        # Collect live data
                        live_data = await self._collect_live_data(session_id)
                        
                        if not live_data:
                            logger.warning("No live data collected")
                            await asyncio.sleep(self.prediction_interval)
                            continue
                        
                        current_lap = live_data.get('current_lap', 1)
                        total_laps = live_data.get('total_laps', 50)
                        
                        logger.info(f"Processing lap {current_lap}/{total_laps}")
                        
                        # Check if we should make final prediction
                        is_final = (total_laps - current_lap) <= self.final_prediction_lap
                        
                        if is_final:
                            logger.info(f"Making FINAL prediction with {total_laps - current_lap} laps to go")
                        
                        # Extract features
                        features_df = self._extract_live_features(live_data, self.current_event)
                        
                        if not features_df.empty:
                            # Make predictions
                            predictions = self._make_predictions(features_df)
                            
                            if predictions:
                                # Save to database
                                self._save_predictions_to_db(
                                    predictions, 
                                    self.current_event, 
                                    None,  # Session object not available
                                    live_data,
                                    is_final
                                )
                                
                                # Log prediction summary
                                if 'catboost' in predictions:
                                    top_5 = np.argsort(predictions['catboost'])[:5]
                                    logger.info(f"Top 5 predicted positions: {top_5}")
                        
                        # If final prediction made, exit
                        if is_final:
                            logger.info("Final prediction completed. Exiting live prediction system.")
                            break
                        
                        # Wait before next prediction
                        await asyncio.sleep(self.prediction_interval)
                        
                    except Exception as e:
                        logger.error(f"Error in prediction loop: {e}")
                        await asyncio.sleep(self.prediction_interval)
                        
        except Exception as e:
            logger.error(f"Error in live prediction system: {e}")

async def main():
    """Main entry point"""
    try:
        prediction_system = LivePredictionSystem()
        await prediction_system.run_live_prediction()
    except KeyboardInterrupt:
        logger.info("Live prediction system stopped by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 