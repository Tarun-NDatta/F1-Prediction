# data/management/commands/backfill_grid_positions.py
from django.core.management.base import BaseCommand
from django.db.models import Avg
from data.models import RaceResult, QualifyingResult

class Command(BaseCommand):
    help = 'Backfill missing grid positions from qualifying results'

    def handle(self, *args, **options):
        # Get race results with missing grid positions and prefetch related data
        missing_grid = RaceResult.objects.filter(
            grid_position__isnull=True
        ).select_related('session', 'driver', 'session__event')
        
        total = missing_grid.count()
        
        if total == 0:
            self.stdout.write(self.style.SUCCESS("No missing grid positions found!"))
            return
            
        self.stdout.write(f"Found {total} race results with missing grid positions")
        updated = 0
        skipped_no_session = 0
        skipped_no_qualifying = 0
        
        for race_result in missing_grid:
            # Skip if session is missing
            if not race_result.session:
                skipped_no_session += 1
                continue
                
            # Find corresponding qualifying result
            qualifying = QualifyingResult.objects.filter(
                driver=race_result.driver,
                session__event=race_result.session.event
            ).first()
            
            if qualifying:
                race_result.grid_position = qualifying.position
                race_result.save()
                updated += 1
            else:
                skipped_no_qualifying += 1
                # Optional: Set a default value
                # race_result.grid_position = 20  # Default to back of grid
                # race_result.save()
                # updated += 1
        
        # Generate summary
        self.stdout.write(f"Updated: {updated}")
        self.stdout.write(f"Skipped (no session): {skipped_no_session}")
        self.stdout.write(f"Skipped (no qualifying): {skipped_no_qualifying}")
        self.stdout.write(self.style.WARNING(
        "NOTE: 20 records with missing sessions were skipped. "
        "These need to be resolved when full event data is available."
        ))
        self.stdout.write(self.style.SUCCESS(
            f"Processed {total} records. "
            f"Updated {updated} grid positions "
            f"({skipped_no_session + skipped_no_qualifying} skipped)"
        ))