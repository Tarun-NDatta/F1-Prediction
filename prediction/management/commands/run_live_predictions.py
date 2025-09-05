"""
Django management command to run live F1 predictions using HypRace API
"""

import asyncio
import os
import sys
import requests
import json
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone
import logging
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

class HypRaceAPIClient:
    """Client for HypRace API"""
    
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
        """Load usage count from file or initialize"""
        try:
            with open('hyprace_usage.json', 'r') as f:
                data = json.load(f)
                current_month = datetime.now().strftime('%Y-%m')
                if data.get('month') == current_month:
                    return data.get('count', 0)
                else:
                    # New month, reset counter
                    return 0
        except FileNotFoundError:
            return 0
    
    def _save_usage_count(self):
        """Save usage count to file"""
        current_month = datetime.now().strftime('%Y-%m')
        data = {
            'month': current_month,
            'count': self.usage_count
        }
        with open('hyprace_usage.json', 'w') as f:
            json.dump(data, f)
    
    def quota_status(self):
        """Get current quota status"""
        current_month = datetime.now().strftime('%Y-%m')
        return {
            'month': current_month,
            'used': self.usage_count,
            'limit': self.monthly_limit,
            'remaining': max(0, self.monthly_limit - self.usage_count)
        }
    
    def _make_request(self, endpoint):
        """Make API request with usage tracking"""
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
        """Test API connection without consuming quota"""
        try:
            # Test DNS resolution
            import socket
            socket.gethostbyname('hyprace-api.p.rapidapi.com')
            
            # Check if API key is valid format (but don't make actual request)
            if len(self.api_key) < 20:
                return False, "API key appears to be invalid format"
            
            return True, "Connection test passed (DNS OK, API key present)"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def get_grands_prix(self):
        """Get list of Grand Prix events"""
        return self._make_request("/v2/grands-prix")
    
    def get_qualifying_results(self, grand_prix_id, session_id):
        """Get qualifying results for specific Grand Prix and session"""
        endpoint = f"/v2/grands-prix/{grand_prix_id}/qualifying/{session_id}/results"
        return self._make_request(endpoint)
    
    def get_race_results(self, grand_prix_id, session_id):
        """Get race results for specific Grand Prix and session"""
        endpoint = f"/v2/grands-prix/{grand_prix_id}/race/{session_id}/results"
        return self._make_request(endpoint)
    
    def get_practice_results(self, grand_prix_id, session_id, practice_number=1):
        """Get practice results for specific Grand Prix and session"""
        endpoint = f"/v2/grands-prix/{grand_prix_id}/practice{practice_number}/{session_id}/results"
        return self._make_request(endpoint)

class LivePredictionSystem:
    """Live prediction system using HypRace API"""
    
    def __init__(self, api_key=None, monthly_limit=40):
        self.client = HypRaceAPIClient(api_key, monthly_limit)
        self.prediction_interval = 10
        self.final_prediction_lap = 15
        
    async def run_predictions(self, grand_prix_id=None, session_id=None):
        """Run live predictions"""
        logger.info("Starting live prediction system with HypRace API")
        
        try:
            # Get available Grand Prix if not specified
            if not grand_prix_id:
                grands_prix = self.client.get_grands_prix()
                logger.info(f"Available Grand Prix: {len(grands_prix)} events")
                
                # Use the first available one for demo
                if grands_prix:
                    grand_prix_id = grands_prix[0].get('id')
                    logger.info(f"Using Grand Prix: {grands_prix[0].get('name', 'Unknown')}")
                else:
                    raise Exception("No Grand Prix events available")
            
            # For demo purposes, using the provided IDs from your curl example
            demo_grand_prix_id = "4c0fc237-e21e-4b9c-b70d-c0d68764e338"
            demo_session_id = "01e0a0e2-04cb-4bd5-80a8-999aa88f764c"
            
            # Get race information
            logger.info("Fetching race data...")
            qualifying_results = self.client.get_qualifying_results(demo_grand_prix_id, demo_session_id)
            
            logger.info(f"Retrieved qualifying data: {len(qualifying_results.get('results', []))} entries")
            
            # Process results for predictions
            await self._process_race_data(qualifying_results)
            
        except Exception as e:
            logger.error(f"Error in prediction system: {e}")
            raise
    
    async def _process_race_data(self, race_data):
        """Process race data and make predictions"""
        logger.info("Processing race data for predictions...")
        
        # Extract driver positions and performance data
        results = race_data.get('results', [])
        
        for result in results[:10]:  # Process top 10
            driver_name = result.get('driver', {}).get('name', 'Unknown')
            position = result.get('position', 'N/A')
            time = result.get('time', 'N/A')
            
            logger.info(f"Position {position}: {driver_name} - {time}")
        
        # Here you would implement your prediction algorithms
        # For now, just log the data structure
        logger.info(f"Race data structure: {json.dumps(race_data, indent=2)}")
    
    def _save_predictions_to_db(self, predictions):
        """Save predictions to database (placeholder)"""
        logger.info(f"Saving {len(predictions)} predictions to database")
        # Implement your database saving logic here

class Command(BaseCommand):
    help = 'Run live F1 prediction system using HypRace API'

    def add_arguments(self, parser):
        parser.add_argument(
            '--interval',
            type=int,
            default=10,
            help='Polling interval in seconds (default: 10)'
        )
        parser.add_argument(
            '--final-lap',
            type=int,
            default=15,
            help='Lap number to make final prediction (default: 15)'
        )
        parser.add_argument(
            '--grand-prix-id',
            type=str,
            help='Specific Grand Prix ID to run predictions for'
        )
        parser.add_argument(
            '--session-id',
            type=str,
            help='Specific session ID to run predictions for'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without saving to database (for testing)'
        )
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level (default: INFO)'
        )
        parser.add_argument(
            '--max-requests',
            type=int,
            default=40,
            help='Monthly request cap for HypRace API usage tracking (default: 40)'
        )
        parser.add_argument(
            '--test-connection',
            action='store_true',
            help='Test API connection without consuming quota'
        )
        parser.add_argument(
            '--quota-status',
            action='store_true',
            help='Show current API usage status and exit'
        )
        parser.add_argument(
            '--list-races',
            action='store_true',
            help='List available Grand Prix events'
        )

    def handle(self, *args, **options):
        # Configure logging
        log_level = getattr(logging, options['log_level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hyprace_prediction.log'),
                logging.StreamHandler()
            ]
        )

        self.stdout.write(
            self.style.SUCCESS('Starting Live F1 Prediction System with HypRace API...')
        )

        try:
            # Create prediction system
            prediction_system = LivePredictionSystem(
                monthly_limit=options['max_requests']
            )

            # Handle quota status check
            if options.get('quota_status'):
                status = prediction_system.client.quota_status()
                self.stdout.write(
                    self.style.SUCCESS(
                        f"HypRace API quota for {status['month']}: "
                        f"used {status['used']}/{status['limit']} "
                        f"(remaining {status['remaining']})"
                    )
                )
                return

            # Handle connection test
            if options.get('test_connection'):
                success, message = prediction_system.client.test_connection()
                style = self.style.SUCCESS if success else self.style.ERROR
                self.stdout.write(style(f"Connection test: {message}"))
                return

            # Handle list races
            if options.get('list_races'):
                try:
                    grands_prix = prediction_system.client.get_grands_prix()
                    self.stdout.write(self.style.SUCCESS(f"Available Grand Prix ({len(grands_prix)} events):"))
                    for gp in grands_prix:
                        self.stdout.write(f"  - {gp.get('name', 'Unknown')} (ID: {gp.get('id', 'N/A')})")
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Failed to fetch Grand Prix list: {e}"))
                return

            # Override settings if provided
            if options['interval']:
                prediction_system.prediction_interval = options['interval']

            if options['final_lap']:
                prediction_system.final_prediction_lap = options['final_lap']

            if options['dry_run']:
                # Modify the system to not save to database
                original_save_method = prediction_system._save_predictions_to_db
                prediction_system._save_predictions_to_db = lambda *args, **kwargs: None
                self.stdout.write(
                    self.style.WARNING('DRY RUN MODE: Predictions will not be saved to database')
                )

            self.stdout.write(
                self.style.SUCCESS(f'Prediction interval: {prediction_system.prediction_interval}s')
            )
            self.stdout.write(
                self.style.SUCCESS(f'Final prediction lap: {prediction_system.final_prediction_lap}')
            )

            # Show current quota status
            status = prediction_system.client.quota_status()
            self.stdout.write(
                self.style.SUCCESS(f"API quota: {status['remaining']} requests remaining this month")
            )

            # Run the prediction system
            asyncio.run(prediction_system.run_predictions(
                grand_prix_id=options.get('grand_prix_id'),
                session_id=options.get('session_id')
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