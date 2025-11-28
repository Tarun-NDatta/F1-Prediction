"""
Place this in: data/management/commands/export_chaos_data.py

Run with: python manage.py export_chaos_data
"""

import os
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from prediction.analysis.chaos import analyze_prediction_errors_by_events, CounterfactualAnalyzer


class Command(BaseCommand):
    help = 'Export chaos analysis data for Streamlit app'

    def handle(self, *args, **options):
        output_dir = './predictor_data'
        os.makedirs(output_dir, exist_ok=True)

        self.stdout.write("Exporting chaos analysis data...")

        try:
            # Run chaos analysis for all seasons
            self.stdout.write("Analyzing prediction errors by events...")
            df = analyze_prediction_errors_by_events(season=None, model_name='catboost_ensemble')
            
            if df.empty:
                self.stdout.write(self.style.WARNING("No chaos analysis data available"))
                return

            # Debug: print columns to see what we have
            self.stdout.write(f"Available columns: {list(df.columns)}")

            # Save raw chaos data
            df.to_csv(f'{output_dir}/chaos_analysis_raw.csv', index=False)
            self.stdout.write(f"  ✓ Exported {len(df)} chaos analysis records")

            # Calculate summary statistics
            cfa = CounterfactualAnalyzer(df)
            
            # Overall metrics
            overall_mae = float(np.mean(np.abs(df['error']))) if not df.empty else 0.0
            clean_mae = cfa.analyze_clean_race_accuracy()
            
            # Counterfactual: perfect chaos knowledge
            df_cf = df.copy()
            if not df_cf.empty and 'driver_affected' in df_cf.columns:
                df_cf.loc[df_cf['driver_affected'] == True, 'error'] = 0.0
                perfect_mae = float(np.mean(np.abs(df_cf['error'])))
            else:
                perfect_mae = overall_mae

            # Category breakdown (check if race_category exists)
            if 'race_category' in df.columns:
                category_stats = []
                for category in df['race_category'].unique():
                    cat_data = df[df['race_category'] == category]
                    category_stats.append({
                        'category': category,
                        'count': len(cat_data),
                        'mae': float(np.mean(np.abs(cat_data['error']))),
                        'affected_drivers': int(cat_data['driver_affected'].sum()) if 'driver_affected' in cat_data.columns else 0,
                    })

                category_df = pd.DataFrame(category_stats)
                category_df.to_csv(f'{output_dir}/chaos_by_category.csv', index=False)
                self.stdout.write(f"  ✓ Exported {len(category_df)} category statistics")
            else:
                self.stdout.write(self.style.WARNING("  ⚠ 'race_category' column not found, skipping category stats"))

            # Event-level summary - use available columns
            # Common column names to try
            event_col = None
            for col in ['event_name', 'event', 'race_name', 'race']:
                if col in df.columns:
                    event_col = col
                    break
            
            year_col = 'year' if 'year' in df.columns else None
            round_col = 'round' if 'round' in df.columns else None
            
            if event_col:
                event_stats = []
                for event_value in df[event_col].unique():
                    event_data = df[df[event_col] == event_value]
                    
                    stat_row = {
                        'event_name': str(event_value),
                        'mae': float(np.mean(np.abs(event_data['error']))),
                        'total_drivers': len(event_data),
                    }
                    
                    # Add optional columns if they exist
                    if year_col and not event_data[year_col].empty:
                        stat_row['year'] = event_data[year_col].iloc[0]
                    
                    if round_col and not event_data[round_col].empty:
                        stat_row['round'] = event_data[round_col].iloc[0]
                    
                    if 'race_category' in event_data.columns:
                        stat_row['race_category'] = event_data['race_category'].iloc[0]
                    else:
                        stat_row['race_category'] = 'unknown'
                    
                    if 'driver_affected' in event_data.columns:
                        stat_row['affected_drivers'] = int(event_data['driver_affected'].sum())
                    else:
                        stat_row['affected_drivers'] = 0
                    
                    event_stats.append(stat_row)

                events_df = pd.DataFrame(event_stats)
                events_df.to_csv(f'{output_dir}/chaos_by_event.csv', index=False)
                self.stdout.write(f"  ✓ Exported {len(events_df)} event statistics")
            else:
                self.stdout.write(self.style.WARNING(f"  ⚠ No event column found. Available columns: {list(df.columns)}"))
                # Create minimal event stats from what we have
                events_df = pd.DataFrame([{
                    'event_name': 'All Races Combined',
                    'mae': overall_mae,
                    'total_drivers': len(df),
                    'race_category': 'mixed',
                    'affected_drivers': int(df['driver_affected'].sum()) if 'driver_affected' in df.columns else 0
                }])
                events_df.to_csv(f'{output_dir}/chaos_by_event.csv', index=False)
                self.stdout.write(f"  ✓ Exported combined event statistics")

            # Summary metrics
            summary = {
                'overall_mae': overall_mae,
                'clean_mae': clean_mae,
                'perfect_mae': perfect_mae,
                'total_predictions': len(df),
                'affected_predictions': int(df['driver_affected'].sum()) if 'driver_affected' in df.columns else 0,
            }
            
            # Add chaos/clean race counts if race_category exists
            if 'race_category' in df.columns:
                summary['chaos_races'] = int((df['race_category'] != 'clean').sum())
                summary['clean_races'] = int((df['race_category'] == 'clean').sum())
            else:
                summary['chaos_races'] = 0
                summary['clean_races'] = len(df)

            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(f'{output_dir}/chaos_summary.csv', index=False)
            self.stdout.write(f"  ✓ Exported summary metrics")

            self.stdout.write(self.style.SUCCESS(f"\n✓ Chaos analysis export complete!"))
            self.stdout.write(f"\nFiles created:")
            self.stdout.write(f"  - chaos_analysis_raw.csv ({len(df)} rows)")
            if 'race_category' in df.columns:
                self.stdout.write(f"  - chaos_by_category.csv ({len(category_df)} rows)")
            self.stdout.write(f"  - chaos_by_event.csv ({len(events_df)} rows)")
            self.stdout.write(f"  - chaos_summary.csv (1 row)")
            
            self.stdout.write(f"\nKey Metrics:")
            self.stdout.write(f"  Overall MAE: {overall_mae:.2f}")
            self.stdout.write(f"  Clean Race MAE: {clean_mae:.2f}")
            self.stdout.write(f"  Perfect Knowledge MAE: {perfect_mae:.2f}")
            self.stdout.write(f"  Chaos Impact: {overall_mae - clean_mae:.2f} positions")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error during export: {str(e)}"))
            import traceback
            self.stdout.write(traceback.format_exc())