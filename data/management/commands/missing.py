# data/management/commands/inspect_missing_sessions.py
from django.core.management.base import BaseCommand
from django.db import transaction
from data.models import RaceResult, Event, Session, SessionType

class Command(BaseCommand):
    help = 'Inspect race results with missing session associations'

    def handle(self, *args, **options):
        # Find race results without sessions
        no_session_results = RaceResult.objects.filter(session__isnull=True)
        count = no_session_results.count()
        
        if count == 0:
            self.stdout.write(self.style.SUCCESS("No race results without sessions found!"))
            return
        
        self.stdout.write(f"Found {count} race results without sessions:")
        
        # Try to find matching sessions for these results
        fixed = 0
        for result in no_session_results:
            # Try to find a matching session
            session = Session.objects.filter(
                event=result.session.event,  # This might be None too
                session_type=SessionType.objects.get(session_type="RACE")
            ).first()
            
            if session:
                result.session = session
                result.save()
                fixed += 1
                self.stdout.write(f"  Fixed result {result.id} → session {session.id}")
            else:
                self.stdout.write(f"  Result {result.id} - No session found for event {result.session.event if result.session else 'N/A'}")
        
        self.stdout.write(self.style.SUCCESS(
            f"Fixed {fixed}/{count} records. {count - fixed} still need attention."
        ))