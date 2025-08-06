"""
Django management command to manage background tasks
"""

from django.core.management.base import BaseCommand
from background_tasks import background_manager, start_background_tasks, stop_background_tasks

class Command(BaseCommand):
    help = 'Manage background tasks for the F1 Dashboard'

    def add_arguments(self, parser):
        parser.add_argument(
            'action',
            choices=['start', 'stop', 'status'],
            help='Action to perform on background tasks'
        )

    def handle(self, *args, **options):
        action = options['action']
        
        if action == 'start':
            self.stdout.write(
                self.style.SUCCESS('Starting background tasks...')
            )
            start_background_tasks()
            self.stdout.write(
                self.style.SUCCESS('Background tasks started successfully!')
            )
            
        elif action == 'stop':
            self.stdout.write(
                self.style.WARNING('Stopping background tasks...')
            )
            stop_background_tasks()
            self.stdout.write(
                self.style.SUCCESS('Background tasks stopped successfully!')
            )
            
        elif action == 'status':
            status = background_manager.get_status()
            self.stdout.write(
                self.style.SUCCESS('Background Tasks Status:')
            )
            self.stdout.write(f"Live Prediction Running: {status['live_prediction_running']}")
            self.stdout.write(f"Live Prediction Thread Alive: {status['live_prediction_thread_alive']}")
            self.stdout.write(f"Last Updated: {status['timestamp']}")
            
            if status['live_prediction_running']:
                self.stdout.write(
                    self.style.SUCCESS('✓ Live prediction system is active')
                )
            else:
                self.stdout.write(
                    self.style.WARNING('✗ Live prediction system is not running')
                ) 