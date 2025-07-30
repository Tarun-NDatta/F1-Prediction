from django.core.management.base import BaseCommand
from data.models import TrackSpecialization
import pandas as pd

class Command(BaseCommand):
    help = 'Analyze TrackSpecialization data'

    def handle(self, *args, **options):
        # Fetch TrackSpecialization data
        track_data = TrackSpecialization.objects.all().values(
            'circuit_id', 'category', 'power_sensitivity', 'aero_sensitivity',
            'tire_degradation_rate', 'overtaking_difficulty', 'qualifying_importance', 'weather_impact'
        )
        df = pd.DataFrame(track_data)

        # Summary statistics
        self.stdout.write("TrackSpecialization Summary:")
        self.stdout.write(str(df.describe()))
        self.stdout.write("\nTrack Category Counts:")
        self.stdout.write(str(df['category'].value_counts()))
        self.stdout.write("\nCircuits 3, 7, 17, 19, 22:")
        self.stdout.write(str(df[df['circuit_id'].isin([3, 7, 17, 19, 22])][['circuit_id', 'category', 'power_sensitivity', 'overtaking_difficulty']]))
        self.stdout.write(self.style.SUCCESS("Analysis completed successfully!"))