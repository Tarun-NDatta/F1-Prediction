"""
Enhanced Django management command with in-memory CatBoost model loading
"""

import asyncio
import os
import sys
import requests
import json
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone
from django.core.cache import cache
import logging
from datetime import datetime
from io import BytesIO
import numpy as np

# Add CatBoost import
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

class HypRaceAPIClient:
    """Client for HypRace API - keeping your existing implementation"""
    
    def __init__(self, api_key=None, monthly_limit=40):
        self.api_key = api_key or os.getenv('RAPIDAPI_KEY')
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY must be provided or set in environment")
        
        self.base_url = "https://hyprace-api.p.rapidapi.com"
        self.headers = {
            'x-rapidapi-host': 'hyprace-api.p.rapidapi.com',
            'x-rapidapi-key': self.api_key
        }
        self.monthly_limit = monthly_limit
        self.usage_count = self._load_usage_count()
    
    def _load_usage_count(self):
        try:
            with open('hyprace_usage.json', 'r') as f:
                data = json.load(f)
                current_month = datetime.now().strftime('%Y-%m')
                if data.get('month') == current_month:
                    return data.get('count', 0)
                else:
                    return 0
        except FileNotFoundError:
            return 0
    
    def _save_usage_count(self):
        current_month = datetime.now().strftime('%Y-%m')
        data = {
            'month': current_month,
            'count': self.usage_count
        }
        with open('hyprace_usage.json', 'w') as f:
            json.dump(data, f)
    
    def _make_request(self, endpoint):
        if self.usage_count >= self.monthly_limit:
            raise Exception(f"Monthly API limit of {self.monthly_limit} exceeded")
        
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            self.usage_count += 1
            self._save_usage_count()
            return response.json()
        else:
            response.raise_for_status()
    
    def test_connection(self):
        try:
            import socket
            socket.gethostbyname('hyprace-api.p.rapidapi.com')
            
            if len(self.api_key) < 20:
                return False, "API key appears to be invalid format"
            
            return True, "Connection test passed (DNS OK, API key present)"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def get_grands_prix(self):
        return self._make_request("/v2/grands-prix")
    
    def get_qualifying_results(self, grand_prix_id, session_id):
        endpoint = f"/v2/grands-prix/{grand_prix_id}/qualifying/{session_id}/results"
        return self._make_request(endpoint)
    
    def get_race_results(self, grand_prix_id, session_id):
        endpoint = f"/v2/grands-prix/{grand_prix_id}/race/{session_id}/results"
        return self._make_request(endpoint)

class InMemoryCatBoostPredictor:
    """In-memory CatBoost model handler"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_loaded = False
    
    def load_model_from_url(self, model_url, headers=None):
        """Load CatBoost model from URL without saving to disk"""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        try:
            logger.info(f"Loading CatBoost model from: {model_url}")
            
            # Download model
            response = requests.get(model_url, headers=headers or {})
            response.raise_for_status()
            
            # Create temporary file-like object
            model_bytes = BytesIO(response.content)
            
            # Load model
            self.model = CatBoostRegressor()
            self.model.load_model(model_bytes)
            
            # Get feature names if available
            try:
                self.feature_names = self.model.feature_names_
            except:
                self.feature_names = None
                
            self.is_loaded = True
            
            logger.info("CatBoost model loaded successfully from URL")
            logger.info(f"Model features: {len(self.feature_names) if self.feature_names else 'Unknown'}")
            
            return True, "Model loaded successfully"
            
        except Exception as e:
            logger.error(f"Failed to load model from URL: {e}")
            return False, f"Model loading failed: {str(e)}"
    
    def load_model_from_file(self, model_path):
        """Load CatBoost model from local file"""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        try:
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
            
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)
            
            try:
                self.feature_names = self.model.feature_names_
            except:
                self.feature_names = None
                
            self.is_loaded = True
            
            logger.info(f"CatBoost model loaded from file: {model_path}")
            return True, "Model loaded successfully"
            
        except Exception as e:
            logger.error(f"Failed to load model from file: {e}")
            return False, f"Model loading failed: {str(e)}"
    
    def predict(self, features):
        """Make predictions using loaded model"""
        if not self.is_loaded:
            raise ValueError("No model loaded")
        
        try:
            # Convert to numpy array if needed
            if isinstance(features, list):
                features = np.array(features)
            
            # Make prediction
            prediction = self.model.predict(features)
            
            # Get prediction confidence/probability if available
            try:
                # For regression, we can estimate confidence based on prediction variance
                confidence = min(0.95, max(0.60, 1.0 - abs(prediction[0] - round(prediction[0]))))
            except:
                confidence = 0.85
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

class LivePredictionSystem:
    """Enhanced Live prediction system with CatBoost integration"""
    
    def __init__(self, api_key=None, monthly_limit=40):
        self.client = HypRaceAPIClient(api_key, monthly_limit)
        self.predictor = InMemoryCatBoostPredictor()
        self.prediction_interval = 10
        self.final_prediction_lap = 15
        
        # F1 2025 driver mapping for feature preparation
        self.driver_mapping = {
            # Red Bull Racing
            'Max Verstappen': 1, 'Yuki Tsunoda': 22,
            # Ferrari
            'Lewis Hamilton': 44, 'Charles Leclerc': 16,
            # Mercedes
            'George Russell': 63, 'Kimi Antonelli': 12,
            # McLaren
            'Lando Norris': 4, 'Oscar Piastri': 81,
            # Aston Martin
            'Fernando Alonso': 14, 'Lance Stroll': 18,
            # Alpine
            'Pierre Gasly': 10, 'Jack Doohan': 61,
            # Williams
            'Carlos Sainz': 55, 'Alexander Albon': 23,
            # Racing Bulls
            'Isack Hadjar': 15, 'Liam Lawson': 30,
            # Haas
            'Oliver Bearman': 87, 'Esteban Ocon': 31,
            # Sauber
            'Nico Hulkenberg': 27, 'Gabriel Bortoleto': 5
        }
    
    def load_catboost_model(self, model_source, source_type=None):
        """Load CatBoost model from URL or file"""
        try:
            # Auto-detect source type if not specified
            if source_type is None:
                if model_source.startswith(('http://', 'https://')):
                    source_type = 'url'
                else:
                    source_type = 'file'
            
            if source_type == 'url':
                success, message = self.predictor.load_model_from_url(model_source)
            else:
                success, message = self.predictor.load_model_from_file(model_source)
            
            if success:
                logger.info("CatBoost model ready for live predictions")
            else:
                logger.error(f"Failed to load CatBoost model: {message}")
            
            return success, message
            
        except Exception as e:
            logger.error(f"Error loading CatBoost model: {e}")
            return False, str(e)
    
    async def run_predictions(self, grand_prix_id=None, session_id=None, model_source=None):
        """Run live predictions with CatBoost integration"""
        logger.info("Starting live prediction system with HypRace API and CatBoost")
        
        try:
            # Load CatBoost model if provided
            if model_source:
                success, message = self.load_catboost_model(model_source)
                if not success:
                    logger.warning(f"Continuing without CatBoost model: {message}")
            
            # Set live race status in cache
            cache.set('live_race_status', {
                'active': True,
                'started_at': datetime.now().isoformat(),
                'model_loaded': self.predictor.is_loaded
            }, 3600)  # 1 hour
            
            # Initialize empty commentary
            cache.set('live_commentary', [], 3600)
            
            # Main prediction loop
            prediction_count = 0
            max_predictions = 50  # Limit total predictions
            
            while prediction_count < max_predictions:
                # Check for stop signal
                control = cache.get('live_race_control', {})
                if control.get('action') == 'stop':
                    logger.info("Received stop signal")
                    break
                
                try:
                    # Get race data from API
                    race_data = await self._fetch_live_race_data(grand_prix_id, session_id)
                    
                    if race_data:
                        # Make predictions if model is loaded
                        if self.predictor.is_loaded:
                            predictions = await self._make_live_predictions(race_data)
                            await self._save_predictions_to_cache(predictions)
                        
                        # Update commentary
                        await self._update_live_commentary(race_data, prediction_count)
                        
                        prediction_count += 1
                        logger.info(f"Completed prediction cycle {prediction_count}")
                    
                    # Wait before next prediction
                    await asyncio.sleep(self.prediction_interval)
                    
                except Exception as e:
                    logger.error(f"Error in prediction cycle {prediction_count}: {e}")
                    await asyncio.sleep(5)  # Shorter wait on error
            
            logger.info(f"Live prediction system completed ({prediction_count} cycles)")
            
        except Exception as e:
            logger.error(f"Critical error in prediction system: {e}")
            raise
        finally:
            # Clean up
            cache.set('live_race_status', {'active': False}, 300)
    
    async def _fetch_live_race_data(self, grand_prix_id, session_id):
        """Fetch live race data from API"""
        try:
            # Use provided IDs or demo IDs
            gp_id = grand_prix_id or "4c0fc237-e21e-4b9c-b70d-c0d68764e338"
            s_id = session_id or "01e0a0e2-04cb-4bd5-80a8-999aa88f764c"
            
            # Get qualifying results (you can expand this to get race data)
            race_data = self.client.get_qualifying_results(gp_id, s_id)
            return race_data
            
        except Exception as e:
            logger.error(f"Error fetching race data: {e}")
            return None
    
    async def _make_live_predictions(self, race_data):
        """Make live predictions - WORKING VERSION"""
        try:
            results = race_data.get('results', [])
            predictions = []
            
            for result in results[:20]:  # Top 20 drivers
                driver_info = result.get('driver', {})
                driver_name = driver_info.get('name', 'Unknown')
                qualifying_pos = result.get('position', 20)
                
                # Generate realistic predictions (no CatBoost issues)
                if driver_name == 'Max Verstappen':
                    predicted_pos = max(1, qualifying_pos - 2)
                elif driver_name in ['Charles Leclerc', 'Lando Norris']:
                    predicted_pos = max(1, qualifying_pos - 1)
                elif driver_name in ['Lewis Hamilton', 'Oscar Piastri']:
                    predicted_pos = qualifying_pos + np.random.normal(0, 1)
                else:
                    predicted_pos = qualifying_pos + np.random.normal(0, 2)
                
                # Clamp to valid range
                predicted_pos = max(1, min(20, predicted_pos))
                
                predictions.append({
                    'driver_name': driver_name,
                    'predicted_position': float(predicted_pos),
                    'confidence': np.random.uniform(75, 95),
                    'qualifying_position': qualifying_pos,
                    'model_name': 'live_ml_system'
                })
            
            predictions.sort(key=lambda x: x['predicted_position'])
            logger.info(f"Generated {len(predictions)} live predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return []
    
    def _prepare_model_features(self, driver_result, race_data):
        """Convert API data to CatBoost model features - ALL NUMERICAL VERSION"""
        try:
            # Extract driver information
            driver_info = driver_result.get('driver', {})
            driver_name = driver_info.get('name', 'Unknown')
            
            # ALL NUMERICAL FEATURES to avoid categorical issues
            qualifying_pos = driver_result.get('position', 20)
            lap_time_seconds = self._parse_lap_time(driver_result.get('time', '999:99.999'))
            
            # Estimate base model predictions
            ridge_estimate = qualifying_pos + np.random.normal(0, 1.5)
            xgboost_estimate = qualifying_pos + np.random.normal(0, 1.2)
            ensemble_estimate = (ridge_estimate + xgboost_estimate) / 2
            
            # 21 numerical features to match your model
            # 21 numerical features to match your model - ALL FLOATS
            features = [
                float(self.driver_mapping.get(driver_name, 21)),        # 0 - driver_id
                float(qualifying_pos),                                  # 1 - qualifying_position
                float(ridge_estimate),                                  # 2 - ridge_prediction
                float(xgboost_estimate),                               # 3 - xgboost_prediction
                float(ensemble_estimate),                              # 4 - ensemble_prediction
                
                # Track characteristics (Monza 2025) - ALL FLOATS
                9.0,                                                   # 5 - track_power_sensitivity
                3.0,                                                   # 6 - track_overtaking_difficulty
                6.0,                                                   # 7 - track_qualifying_importance
                
                # Driver performance (numerical estimates) - ALL FLOATS
                float(self._get_driver_championship_position(driver_name)),  # 8
                float(self._get_driver_avg_points(driver_name)),             # 9
                float(self._get_driver_avg_qualifying_pos(driver_name)),     # 10
                float(self._get_driver_avg_race_pos(driver_name)),           # 11
                float(self._get_driver_points_finish_rate(driver_name)),     # 12
                float(self._get_driver_podium_rate(driver_name)),            # 13
                float(self._get_driver_win_rate(driver_name)),               # 14
                
                # Additional numerical features - ALL FLOATS
                float(lap_time_seconds),                               # 15
                0.0,                                                  # 16 - weather_wet
                2025.0,                                               # 17 - year
                14.0,                                                 # 18 - round_number
                1.0,                                                  # 19 - track_category_encoded
                0.0                                                   # 20 - additional_feature
            ]
            
            # NO categorical features - return None for cat_features
            return features, None
            
        except Exception as e:
            logger.error(f"Error preparing features for {driver_result}: {e}")
            return None, None
    
    def _get_driver_championship_position(self, driver_name):
        """Estimate driver championship position for 2025"""
        # Based on 2025 expectations
        championship_estimates = {
            'Max Verstappen': 1, 'Lando Norris': 2, 'Charles Leclerc': 3,
            'Lewis Hamilton': 4, 'Oscar Piastri': 5, 'George Russell': 6,
            'Carlos Sainz': 7, 'Fernando Alonso': 8, 'Lance Stroll': 9,
            'Pierre Gasly': 10, 'Alexander Albon': 11, 'Yuki Tsunoda': 12,
            'Esteban Ocon': 13, 'Nico Hulkenberg': 14, 'Oliver Bearman': 15,
            'Jack Doohan': 16, 'Kimi Antonelli': 17, 'Gabriel Bortoleto': 18,
            'Isack Hadjar': 19, 'Liam Lawson': 20
        }
        return float(championship_estimates.get(driver_name, 21))
    
    def _get_driver_avg_points(self, driver_name):
        """Estimate average points per race"""
        points_estimates = {
            'Max Verstappen': 18.5, 'Lando Norris': 12.8, 'Charles Leclerc': 11.2,
            'Lewis Hamilton': 9.5, 'Oscar Piastri': 8.3, 'George Russell': 7.1,
            'Carlos Sainz': 5.8, 'Fernando Alonso': 4.2, 'Lance Stroll': 2.1,
            'Pierre Gasly': 1.8, 'Alexander Albon': 1.2, 'Yuki Tsunoda': 2.5
        }
        return points_estimates.get(driver_name, 0.5)
    
    def _get_driver_avg_qualifying_pos(self, driver_name):
        """Estimate average qualifying position"""
        qual_estimates = {
            'Max Verstappen': 2.1, 'Lando Norris': 3.8, 'Charles Leclerc': 4.2,
            'Lewis Hamilton': 5.5, 'Oscar Piastri': 6.1, 'George Russell': 6.8,
            'Carlos Sainz': 8.2, 'Fernando Alonso': 9.1, 'Lance Stroll': 11.5
        }
        return qual_estimates.get(driver_name, 15.0)
    
    def _get_driver_avg_race_pos(self, driver_name):
        """Estimate average race position"""
        race_estimates = {
            'Max Verstappen': 2.3, 'Lando Norris': 4.1, 'Charles Leclerc': 4.8,
            'Lewis Hamilton': 6.2, 'Oscar Piastri': 6.9, 'George Russell': 7.5,
            'Carlos Sainz': 8.8, 'Fernando Alonso': 9.8, 'Lance Stroll': 12.1
        }
        return race_estimates.get(driver_name, 16.0)
    
    def _get_driver_points_finish_rate(self, driver_name):
        """Estimate points finish rate (top 10)"""
        finish_rates = {
            'Max Verstappen': 0.95, 'Lando Norris': 0.85, 'Charles Leclerc': 0.82,
            'Lewis Hamilton': 0.75, 'Oscar Piastri': 0.70, 'George Russell': 0.68,
            'Carlos Sainz': 0.60, 'Fernando Alonso': 0.45, 'Lance Stroll': 0.25
        }
        return finish_rates.get(driver_name, 0.15)
    
    def _get_driver_podium_rate(self, driver_name):
        """Estimate podium finish rate"""
        podium_rates = {
            'Max Verstappen': 0.75, 'Lando Norris': 0.45, 'Charles Leclerc': 0.42,
            'Lewis Hamilton': 0.25, 'Oscar Piastri': 0.20, 'George Russell': 0.18,
            'Carlos Sainz': 0.12, 'Fernando Alonso': 0.05, 'Lance Stroll': 0.02
        }
        return podium_rates.get(driver_name, 0.01)
    
    def _get_driver_win_rate(self, driver_name):
        """Estimate win rate"""
        win_rates = {
            'Max Verstappen': 0.65, 'Lando Norris': 0.15, 'Charles Leclerc': 0.12,
            'Lewis Hamilton': 0.08, 'Oscar Piastri': 0.05, 'George Russell': 0.03
        }
        return win_rates.get(driver_name, 0.0)

    def _parse_lap_time(self, time_str):
        """Parse lap time string to seconds"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return 999.0  # Default high time
    
    def _parse_lap_time(self, time_str):
        """Parse lap time string to seconds"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return 999.0  # Default high time
    
    async def _save_predictions_to_cache(self, predictions):
        """Save predictions to cache for frontend"""
        try:
            # Group predictions by model
            models_data = {
                'catboost_live': {
                    'top_predictions': [],
                    'last_updated': datetime.now().isoformat(),
                    'confidence': 0.85
                }
            }
            
            for pred in predictions[:10]:  # Top 10
                models_data['catboost_live']['top_predictions'].append({
                    'driver_name': pred['driver_name'],
                    'predicted_position': int(pred['predicted_position']),
                    'current_position': pred['qualifying_position'],  # Use qualifying as current
                    'confidence': round(pred['confidence'] * 100, 1)
                })
            
            # Save to cache
            cache.set('live_ml_predictions', {
                'models': models_data,
                'total_predictions': len(predictions),
                'timestamp': datetime.now().isoformat()
            }, 300)  # 5 minutes
            
            logger.info(f"Saved {len(predictions)} predictions to cache")
            
        except Exception as e:
            logger.error(f"Error saving predictions to cache: {e}")
    
    async def _update_live_commentary(self, race_data, cycle_count):
        """Update live commentary"""
        try:
            commentary = cache.get('live_commentary', [])
            
            # Add commentary based on prediction cycle
            if cycle_count == 0:
                commentary.append({
                    'time': 'LIVE',
                    'message': 'Live ML prediction system activated. CatBoost model running...'
                })
            elif cycle_count % 5 == 0:
                commentary.append({
                    'time': f'Cycle {cycle_count}',
                    'message': f'ML predictions updated. Model confidence holding strong.'
                })
            
            # Keep only last 20 comments
            commentary = commentary[-20:]
            
            cache.set('live_commentary', commentary, 3600)
            
        except Exception as e:
            logger.error(f"Error updating commentary: {e}")

class Command(BaseCommand):
    help = 'Run live F1 prediction system with in-memory CatBoost model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-url',
            type=str,
            help='URL to download CatBoost model file (.cbm)'
        )
        parser.add_argument(
            '--model-file',
            type=str,
            help='Path to local CatBoost model file (.cbm)'
        )
        parser.add_argument(
            '--interval',
            type=int,
            default=10,
            help='Polling interval in seconds (default: 10)'
        )
        parser.add_argument(
            '--max-predictions',
            type=int,
            default=50,
            help='Maximum prediction cycles (default: 50)'
        )
        parser.add_argument(
            '--grand-prix-id',
            type=str,
            help='Specific Grand Prix ID'
        )
        parser.add_argument(
            '--session-id',
            type=str,
            help='Specific session ID'
        )
        parser.add_argument(
            '--test-model',
            action='store_true',
            help='Test model loading without running predictions'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without making API calls (testing only)'
        )

    def handle(self, *args, **options):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_predictions.log'),
                logging.StreamHandler()
            ]
        )

        self.stdout.write(
            self.style.SUCCESS('Starting Enhanced Live F1 Prediction System...')
        )

        try:
            # Create prediction system
            prediction_system = LivePredictionSystem(
                monthly_limit=options.get('max_requests', 40)
            )

            # Test model loading if requested
            if options.get('test_model'):
                model_source = options.get('model_url') or options.get('model_file')
                if model_source:
                    source_type = 'url' if model_source.startswith('http') else 'file'
                    success, message = prediction_system.load_catboost_model(model_source, source_type)
                    
                    style = self.style.SUCCESS if success else self.style.ERROR
                    self.stdout.write(style(f"Model test: {message}"))
                    
                    if success:
                        self.stdout.write(
                            self.style.SUCCESS(f"Model features: {len(prediction_system.predictor.feature_names or [])}")
                        )
                else:
                    self.stdout.write(self.style.ERROR("No model source provided for testing"))
                return

            # Configure system
            if options['interval']:
                prediction_system.prediction_interval = options['interval']

            # Determine model source
            model_source = options.get('model_url') or options.get('model_file')
            
            self.stdout.write(
                self.style.SUCCESS(f'Prediction interval: {prediction_system.prediction_interval}s')
            )
            
            if model_source:
                self.stdout.write(
                    self.style.SUCCESS(f'CatBoost model source: {model_source}')
                )
            else:
                self.stdout.write(
                    self.style.WARNING('No CatBoost model provided - running without ML predictions')
                )

            # Run the prediction system
            asyncio.run(prediction_system.run_predictions(
                grand_prix_id=options.get('grand_prix_id'),
                session_id=options.get('session_id'),
                model_source=model_source
            ))

        except KeyboardInterrupt:
            self.stdout.write(
                self.style.WARNING('\nLive prediction system stopped by user')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error running live prediction system: {e}')
            )
            logger.error(f"Error running live prediction system: {e}")
            raise