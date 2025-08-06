"""
Background Task System for Live F1 Predictions
Runs live predictions in the background while Django server is running
"""

import asyncio
import threading
import time
import logging
from django.conf import settings
from django.utils import timezone
from live_prediction_system import LivePredictionSystem

logger = logging.getLogger('live_prediction')

class BackgroundTaskManager:
    """Manages background tasks for the Django application"""
    
    def __init__(self):
        self.live_prediction_task = None
        self.live_prediction_thread = None
        self.is_running = False
        self.prediction_system = None
    
    def start_live_prediction_task(self):
        """Start the live prediction task in a background thread"""
        if self.is_running:
            logger.info("Live prediction task is already running")
            return
        
        try:
            # Create prediction system
            self.prediction_system = LivePredictionSystem()
            
            # Get configuration from Django settings
            interval = getattr(settings, 'BACKGROUND_TASKS', {}).get('LIVE_PREDICTION_INTERVAL', 30)
            final_lap = getattr(settings, 'BACKGROUND_TASKS', {}).get('LIVE_PREDICTION_FINAL_LAP', 15)
            
            # Override settings
            self.prediction_system.prediction_interval = interval
            self.prediction_system.final_prediction_lap = final_lap
            
            # Start in background thread
            self.live_prediction_thread = threading.Thread(
                target=self._run_live_prediction_async,
                daemon=True
            )
            self.live_prediction_thread.start()
            
            self.is_running = True
            logger.info(f"Live prediction task started (interval: {interval}s, final lap: {final_lap})")
            
        except Exception as e:
            logger.error(f"Error starting live prediction task: {e}")
    
    def stop_live_prediction_task(self):
        """Stop the live prediction task"""
        if not self.is_running:
            logger.info("Live prediction task is not running")
            return
        
        try:
            self.is_running = False
            if self.prediction_system:
                # Signal the prediction system to stop
                self.prediction_system.should_stop = True
            
            logger.info("Live prediction task stopped")
            
        except Exception as e:
            logger.error(f"Error stopping live prediction task: {e}")
    
    def _run_live_prediction_async(self):
        """Run the live prediction system in an async event loop"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the prediction system
            loop.run_until_complete(self.prediction_system.run_live_prediction())
            
        except Exception as e:
            logger.error(f"Error in live prediction async task: {e}")
        finally:
            try:
                loop.close()
            except:
                pass
    
    def get_status(self):
        """Get the status of background tasks"""
        return {
            'live_prediction_running': self.is_running,
            'live_prediction_thread_alive': self.live_prediction_thread.is_alive() if self.live_prediction_thread else False,
            'timestamp': timezone.now().isoformat()
        }

# Global instance
background_manager = BackgroundTaskManager()

def start_background_tasks():
    """Start all background tasks"""
    try:
        # Check if live prediction is enabled
        if getattr(settings, 'BACKGROUND_TASKS', {}).get('LIVE_PREDICTION_ENABLED', False):
            background_manager.start_live_prediction_task()
        else:
            logger.info("Live prediction is disabled in settings")
            
    except Exception as e:
        logger.error(f"Error starting background tasks: {e}")

def stop_background_tasks():
    """Stop all background tasks"""
    try:
        background_manager.stop_live_prediction_task()
    except Exception as e:
        logger.error(f"Error stopping background tasks: {e}")

def get_background_status():
    """Get status of all background tasks"""
    return background_manager.get_status() 