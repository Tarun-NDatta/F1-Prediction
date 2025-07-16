from django.core.management.base import BaseCommand
from prediction.data_prep.pipeline import F1DataPipeline
from data.models import RaceResult
import pandas as pd

class Command(BaseCommand):
    help = 'Run data loading diagnostics'

    def handle(self, *args, **options):
        pipeline = F1DataPipeline(impute_strategy='drop')
        
        try:
            self.stdout.write("Loading data...")
            df = pipeline.load_data()
            
            self.stdout.write("\n=== Data Shape ===")
            self.stdout.write(f"Records: {len(df)}")
            self.stdout.write(f"Features: {len(df.columns)}")
            
            self.stdout.write("\n=== Missing Values ===")
            missing = df.isnull().sum()
            self.stdout.write(missing[missing > 0].to_string())
            
            self.stdout.write("\n=== Position Distribution ===")
            self.stdout.write(df['position'].value_counts().sort_index().to_string())
            
            self.stdout.write("\n=== First 5 Records ===")
            self.stdout.write(df.head().to_string())
            self.stdout.write("\n=== Data Completeness ===")
            self.stdout.write(f"Race results without sessions: {RaceResult.objects.filter(session__isnull=True).count()}")
            self.stdout.write(f"Race results without grid positions: {RaceResult.objects.filter(grid_position__isnull=True).count()}")
            
            self.stdout.write(self.style.SUCCESS("\nDiagnostics completed!"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))