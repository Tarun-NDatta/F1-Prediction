from django.apps import AppConfig


class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dashboard'
    
    def ready(self):
        """Start background tasks when Django is ready"""
        import os
        if os.environ.get('RUN_MAIN', None) != 'true':
            # Only start background tasks once (avoid duplicate in development)
            try:
                from background_tasks import start_background_tasks
                start_background_tasks()
            except Exception as e:
                print(f"Could not start background tasks: {e}")
