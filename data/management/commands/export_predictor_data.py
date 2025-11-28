"""
Place this in: data/management/commands/export_predictor_data.py

Run with: python manage.py export_predictor_data
"""

import os
import pandas as pd
from django.core.management.base import BaseCommand
from data.models import (
    Event, Driver, Circuit, TrackSpecialization,
    ridgeregression, xgboostprediction, CatBoostPrediction
)


class Command(BaseCommand):
    help = 'Export minimal data needed for Streamlit predictor app'

    def handle(self, *args, **options):
        output_dir = './predictor_data'
        os.makedirs(output_dir, exist_ok=True)

        self.stdout.write("Exporting predictor data...")

        # 1. Export 2025 events (upcoming races)
        self.stdout.write("Exporting 2025 events...")
        events = Event.objects.filter(year=2025).select_related('circuit').order_by('round')
        events_data = []
        
        for event in events:
            events_data.append({
                'event_id': event.id,
                'year': event.year,
                'round': event.round,
                'name': event.name,
                'circuit_name': event.circuit.name,
                'circuit_country': event.circuit.country,
                'date': event.date.isoformat() if event.date else None,
            })
        
        events_df = pd.DataFrame(events_data)
        events_df.to_csv(f'{output_dir}/events_2025.csv', index=False)
        self.stdout.write(f"  ✓ Exported {len(events_df)} events")

        # 2. Export current drivers (2025 season)
        self.stdout.write("Exporting 2025 drivers...")
        
        # Get drivers who have predictions in 2025
        driver_ids = set()
        for model_cls in [ridgeregression, xgboostprediction, CatBoostPrediction]:
            preds = model_cls.objects.filter(year=2025).values_list('driver_id', flat=True).distinct()
            driver_ids.update(preds)
        
        drivers = Driver.objects.filter(id__in=driver_ids)
        drivers_data = []
        
        for driver in drivers:
            drivers_data.append({
                'driver_id': driver.id,
                'given_name': driver.given_name,
                'family_name': driver.family_name,
                'full_name': f"{driver.given_name} {driver.family_name}",
                'code': driver.code,
                'permanent_number': driver.permanent_number,
                'nationality': driver.nationality,
            })
        
        drivers_df = pd.DataFrame(drivers_data)
        drivers_df.to_csv(f'{output_dir}/drivers_2025.csv', index=False)
        self.stdout.write(f"  ✓ Exported {len(drivers_df)} drivers")

        # 3. Export track specializations
        self.stdout.write("Exporting track characteristics...")
        
        circuits = Circuit.objects.filter(event__year=2025).distinct()
        track_data = []
        
        for circuit in circuits:
            try:
                track_spec = TrackSpecialization.objects.get(circuit=circuit)
                track_data.append({
                    'circuit_id': circuit.id,
                    'circuit_name': circuit.name,
                    'track_category': track_spec.category,
                    'power_sensitivity': float(track_spec.power_sensitivity or 0),
                    'overtaking_difficulty': float(track_spec.overtaking_difficulty or 0),
                    'qualifying_importance': float(track_spec.qualifying_importance or 0),
                })
            except TrackSpecialization.DoesNotExist:
                track_data.append({
                    'circuit_id': circuit.id,
                    'circuit_name': circuit.name,
                    'track_category': 'Balanced',
                    'power_sensitivity': 5.0,
                    'overtaking_difficulty': 5.0,
                    'qualifying_importance': 5.0,
                })
        
        track_df = pd.DataFrame(track_data)
        track_df.to_csv(f'{output_dir}/track_specs.csv', index=False)
        self.stdout.write(f"  ✓ Exported {len(track_df)} track characteristics")

        # 4. Export predictions for all 3 models (2025 only)
        self.stdout.write("Exporting model predictions...")
        
        all_predictions = []
        
        # Ridge predictions
        ridge_preds = ridgeregression.objects.filter(year=2025).select_related('driver', 'event')
        for pred in ridge_preds:
            all_predictions.append({
                'event_id': pred.event_id,
                'event_name': pred.event.name,
                'round': pred.round_number,
                'driver_id': pred.driver_id,
                'driver_name': f"{pred.driver.given_name} {pred.driver.family_name}",
                'model': 'Ridge',
                'predicted_position': float(pred.predicted_position),
                'actual_position': float(pred.actual_position) if pred.actual_position else None,
            })
        
        # XGBoost predictions
        xgb_preds = xgboostprediction.objects.filter(year=2025).select_related('driver', 'event')
        for pred in xgb_preds:
            all_predictions.append({
                'event_id': pred.event_id,
                'event_name': pred.event.name,
                'round': pred.round_number,
                'driver_id': pred.driver_id,
                'driver_name': f"{pred.driver.given_name} {pred.driver.family_name}",
                'model': 'XGBoost',
                'predicted_position': float(pred.predicted_position),
                'actual_position': float(pred.actual_position) if pred.actual_position else None,
            })
        
        # CatBoost predictions
        catboost_preds = CatBoostPrediction.objects.filter(year=2025).select_related('driver', 'event')
        for pred in catboost_preds:
            all_predictions.append({
                'event_id': pred.event_id,
                'event_name': pred.event.name,
                'round': pred.round_number,
                'driver_id': pred.driver_id,
                'driver_name': f"{pred.driver.given_name} {pred.driver.family_name}",
                'model': 'CatBoost',
                'predicted_position': float(pred.predicted_position),
                'actual_position': float(pred.actual_position) if pred.actual_position else None,
            })
        
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(f'{output_dir}/predictions_2025.csv', index=False)
        self.stdout.write(f"  ✓ Exported {len(predictions_df)} predictions")

        # 5. Create summary file
        summary = {
            'events': len(events_df),
            'drivers': len(drivers_df),
            'tracks': len(track_df),
            'predictions': len(predictions_df),
            'exported_at': pd.Timestamp.now().isoformat(),
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f'{output_dir}/export_summary.csv', index=False)

        self.stdout.write(self.style.SUCCESS(f"\n✓ Export complete! Data saved to: {output_dir}"))
        self.stdout.write(f"\nFiles created:")
        self.stdout.write(f"  - events_2025.csv ({len(events_df)} rows)")
        self.stdout.write(f"  - drivers_2025.csv ({len(drivers_df)} rows)")
        self.stdout.write(f"  - track_specs.csv ({len(track_df)} rows)")
        self.stdout.write(f"  - predictions_2025.csv ({len(predictions_df)} rows)")
        self.stdout.write(f"\nNext step: Copy these files to your Streamlit app's data/ folder")