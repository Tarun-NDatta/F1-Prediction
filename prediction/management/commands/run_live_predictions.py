"""
Django management command to run live F1 predictions
"""

import asyncio
import os
import sys
from django.core.management.base import BaseCommand
from django.conf import settings
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from live_prediction_system import LivePredictionSystem

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run live F1 prediction system using OpenF1 API'

    def add_arguments(self, parser):
        parser.add_argument(
            '--interval',
            type=int,
            default=10,
            help='Polling interval in seconds for RapidAPI (default: 10)'
        )
        parser.add_argument(
            '--final-lap',
            type=int,
            default=15,
            help='Lap number to make final prediction (default: 15)'
        )
        parser.add_argument(
            '--event-id',
            type=int,
            help='Specific event ID to run predictions for'
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
            '--max-requests', type=int, default=40,
            help='Monthly request cap for RapidAPI usage tracking (default: 40)'
        )
        parser.add_argument(
            '--test-connection', action='store_true',
            help='Validate API key and connectivity without consuming quota'
        )
        parser.add_argument(
            '--quota-status', action='store_true',
            help='Print RapidAPI usage (used/remaining) and exit'
        )

    def handle(self, *args, **options):
        # Configure logging
        log_level = getattr(logging, options['log_level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_prediction.log'),
                logging.StreamHandler()
            ]
        )

        self.stdout.write(
            self.style.SUCCESS('Starting Live F1 Prediction System...')
        )

        try:
            # Create prediction system
            prediction_system = LivePredictionSystem()

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

            # RapidAPI-only now
            prediction_system.prediction_interval = options.get('interval') or 10
            # Set the RapidAPI monthly limit for safety
            prediction_system.rapidapi_monthly_limit = options.get('max_requests') or 40

            if options.get('quota_status'):
                # Inspect quota without making a network call
                from live_prediction_system import RapidAPIClient
                client = RapidAPIClient(monthly_limit=prediction_system.rapidapi_monthly_limit)
                status = client.quota_status()
                self.stdout.write(self.style.SUCCESS(f"RapidAPI quota {status['month']}: used {status['used']}/{status['limit']} (remaining {status['remaining']})"))
                return

            if options.get('test_connection'):
                # Lightweight test: verify key presence and DNS without consuming quota
                api_key = os.getenv('RAPIDAPI_KEY')
                if not api_key:
                    raise RuntimeError('RAPIDAPI_KEY not set in environment/.env')
                import socket
                try:
                    socket.gethostbyname('f1-live-pulse.p.rapidapi.com')
                    self.stdout.write(self.style.SUCCESS('DNS resolution OK; API key present. Skipping request to preserve quota.'))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'DNS check failed: {e}'))
                return

            asyncio.run(prediction_system.run_rapidapi_poll(prediction_system.prediction_interval))

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