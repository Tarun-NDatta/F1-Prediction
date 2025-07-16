# prediction/management/commands/check_sessions.py
from django.core.management.base import BaseCommand
from data.models import Session, SessionType, RaceResult

class Command(BaseCommand):
    help = 'Check session data'

    def handle(self, *args, **options):
        # Check available session types
        self.stdout.write("=== Session Types ===")
        for st in SessionType.objects.all():
            self.stdout.write(f"{st.id}: {st.name} ({st.session_type})")

        # Check session counts
        self.stdout.write("\n=== Session Counts ===")
        self.stdout.write(f"Total Sessions: {Session.objects.count()}")
        for st in SessionType.objects.all():
            count = Session.objects.filter(session_type=st).count()
            self.stdout.write(f"{st.session_type}: {count} sessions")

        # Check race results
        self.stdout.write("\n=== Race Results ===")
        self.stdout.write(f"Total RaceResult records: {RaceResult.objects.count()}")
        self.stdout.write(f"RaceResults with position: {RaceResult.objects.filter(position__isnull=False).count()}")
        
        # Check if any race results are linked to Race sessions
        race_sessions = Session.objects.filter(session_type__session_type='Race')
        self.stdout.write(f"\nRace sessions found: {race_sessions.count()}")
        if race_sessions.exists():
            race_results = RaceResult.objects.filter(session__in=race_sessions, position__isnull=False)
            self.stdout.write(f"RaceResults in Race sessions with position: {race_results.count()}")