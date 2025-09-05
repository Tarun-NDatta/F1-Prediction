# background_tasks.py - Background task runner for live race predictions

import threading
import subprocess
import os
import signal
import time
import logging
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)

class LiveRaceTaskManager:
    """Manages background execution of the live race prediction command"""
    
    def __init__(self):
        self.process = None
        self.is_running = False
        self.thread = None
    
    def start_live_race_command(self):
        """Start the Django management command in background"""
        try:
            if self.is_running:
                logger.warning("Live race command already running")
                return False, "Live race system already active"
            
            # Set control signal for starting
            cache.set('live_race_control', {'action': 'start'}, 300)
            
            # Build command
            cmd = [
                'python', 
                'manage.py', 
                'run_live_predictions',
                '--log-level', 'INFO',
                '--max-requests', '40'
            ]
            
            # Start process in background
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=settings.BASE_DIR
            )
            
            self.is_running = True
            
            # Start monitoring thread
            self.thread = threading.Thread(target=self._monitor_process)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info(f"Started live race command with PID: {self.process.pid}")
            
            return True, f"Live race system started (PID: {self.process.pid})"
            
        except Exception as e:
            logger.error(f"Error starting live race command: {e}")
            return False, f"Error starting live race system: {str(e)}"
    
    def stop_live_race_command(self):
        """Stop the running Django management command"""
        try:
            if not self.is_running or not self.process:
                logger.warning("No live race command running")
                return False, "No live race system running"
            
            # Set stop signal
            cache.set('live_race_control', {'action': 'stop'}, 300)
            
            # Give process time to gracefully shut down
            time.sleep(2)
            
            # Force terminate if still running
            if self.process.poll() is None:
                self.process.terminate()
                
                # Wait for termination
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if terminate didn't work
                    self.process.kill()
                    self.process.wait()
            
            self.is_running = False
            logger.info("Live race command stopped")
            
            # Clear cache
            cache.delete('live_race_status')
            cache.delete('live_commentary')
            cache.delete('live_race_progress')
            
            return True, "Live race system stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping live race command: {e}")
            return False, f"Error stopping live race system: {str(e)}"
    
    def _monitor_process(self):
        """Monitor the background process"""
        try:
            while self.is_running and self.process:
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process has ended
                    self.is_running = False
                    
                    # Get exit code and output
                    stdout, stderr = self.process.communicate()
                    exit_code = self.process.returncode
                    
                    if exit_code == 0:
                        logger.info("Live race command completed successfully")
                    else:
                        logger.error(f"Live race command failed with exit code {exit_code}")
                        if stderr:
                            logger.error(f"STDERR: {stderr}")
                    
                    # Clear race status
                    cache.delete('live_race_status')
                    break
                
                # Sleep before next check
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error monitoring live race process: {e}")
            self.is_running = False
    
    def get_status(self):
        """Get current status of live race system"""
        if self.is_running and self.process:
            return {
                'running': True,
                'pid': self.process.pid,
                'uptime': time.time() - self.process.args[0] if hasattr(self.process, 'args') else 'Unknown'
            }
        else:
            return {
                'running': False,
                'pid': None,
                'uptime': None
            }

# Global instance
live_race_manager = LiveRaceTaskManager()

# Updated views to use the task manager
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json

@method_decorator(csrf_exempt, name='dispatch')
class LiveRaceControlView(View):
    """Enhanced view to control live race data collection with background tasks"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            action = data.get('action')
            
            if action == 'start':
                success, message = live_race_manager.start_live_race_command()
                
                return JsonResponse({
                    'success': success,
                    'message': message,
                    'status': live_race_manager.get_status()
                })
            
            elif action == 'stop':
                success, message = live_race_manager.stop_live_race_command()
                
                return JsonResponse({
                    'success': success,
                    'message': message
                })
            
            elif action == 'status':
                return JsonResponse({
                    'success': True,
                    'status': live_race_manager.get_status(),
                    'cache_status': {
                        'race_active': bool(cache.get('live_race_status', {}).get('active', False)),
                        'commentary_count': len(cache.get('live_commentary', [])),
                        'has_progress': bool(cache.get('live_race_progress'))
                    }
                })
            
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid action. Use: start, stop, or status'
                }, status=400)
                
        except Exception as e:
            logger.error(f"Error in LiveRaceControlView: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

# Add this to your Django settings.py
"""
# Background task settings
LIVE_RACE_TASK_TIMEOUT = 3600  # 1 hour
LIVE_RACE_LOG_LEVEL = 'INFO'
LIVE_RACE_MAX_PREDICTIONS = 1000  # Maximum predictions to store

# Cache settings for live data
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Or use database cache if you don't have Redis
# CACHES = {
#     'default': {
#         'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
#         'LOCATION': 'cache_table',
#     }
# }

# Don't forget to run: python manage.py createcachetable
"""