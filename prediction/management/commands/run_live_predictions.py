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
            default=30,
            help='Prediction interval in seconds (default: 30)'
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

            # Run the prediction system
            asyncio.run(prediction_system.run_live_prediction())

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