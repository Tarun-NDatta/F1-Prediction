# dissertation/management/commands/analyze_chaos.py

from django.core.management.base import BaseCommand
from django.db.models import F, Func, Value, FloatField
from django.db.models.functions import Cast
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from data.models import Event, RaceResult, CatBoostPrediction
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive 'Agg' backend to avoid font logging
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

class Command(BaseCommand):
    help = 'Performs chaos analysis: Calculates a Chaos Score for each race and runs a regression against model MAE.'

    def add_arguments(self, parser):
        # Optional argument to specify a year, defaults to 2025
        parser.add_argument(
            '--year',
            type=int,
            default=2025,
            help='The championship year to analyze (default: 2025).',
        )
        parser.add_argument(
            '--model',
            type=str,
            default='catboost_ensemble',
            help='The prediction model to evaluate (default: catboost_ensemble).',
        )

    def calculate_chaos_score_for_event(self, event):
        """
        Calculates a chaos score (0-10) for a given Event based on:
        1. DNF Rate (0-4 points)
        2. Variance in Position Gain (0-3 points)
        3. Variance in Pit Stops (0-3 points)
        """
        # Get all race results for this event
        results = RaceResult.objects.filter(session__event=event).select_related('driver')
        
        if not results.exists():
            return None

        total_drivers = results.count()
        
        # 1. Calculate DNF Score (0-4 points)
        dnf_count = results.exclude(status='Finished').count()
        dnf_score = (dnf_count / total_drivers) * 4  # Max 4 points

        # 2. Calculate Position Change Variance Score (0-3 points)
        # First, get position gains (grid - finish). Filter out DNFs for a cleaner signal.
        finished_results = results.filter(status='Finished')
        position_gains = []
        for result in finished_results:
            if result.grid_position is not None and result.position is not None:
                gain = result.grid_position - result.position  # +ve = gained positions
                position_gains.append(gain)
        
        if position_gains:
            pos_gain_variance = np.var(position_gains)
            pos_gain_score = min(3.0, pos_gain_variance / 10.0) # Cap at 3 points
        else:
            pos_gain_score = 0.0

        # 3. Calculate Pit Stop Variance Score (0-3 points)
        pit_stops_list = list(results.exclude(pit_stops__isnull=True).values_list('pit_stops', flat=True))
        if pit_stops_list:
            pit_stop_variance = np.var(pit_stops_list)
            pit_stop_score = min(3.0, pit_stop_variance * 1.5) # Cap at 3 points
        else:
            pit_stop_score = 0.0

        total_chaos_score = dnf_score + pos_gain_score + pit_stop_score
        # Normalize slightly if needed (though 10 is a reasonable max)
        return min(10.0, total_chaos_score)

    def calculate_mae_for_event(self, event, model_name):
        """
        Calculates the Mean Absolute Error (MAE) for a given Event and prediction model.
        """
        # This query joins predictions with actual race results for the event
        # and calculates the absolute error for each driver.
        predictions_with_error = CatBoostPrediction.objects.filter(
            event=event,
            model_name=model_name,
            actual_position__isnull=False  # Ensure we only look at completed races
        ).annotate(
            absolute_error=Func(
                Cast(F('predicted_position'), output_field=FloatField()) - 
                Cast(F('actual_position'), output_field=FloatField()),
                function='ABS'
            )
        )

        if not predictions_with_error.exists():
            return None

        # Calculate the average of all absolute errors for the race -> MAE
        total_error = sum([p.absolute_error for p in predictions_with_error])
        mae = total_error / predictions_with_error.count()
        
        return mae

    def handle(self, *args, **options):
        year = options['year']
        model_name = options['model']

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Calculating Chaos Theory: Race Chaos vs. Prediction Error for {year}"
        ))
        self.stdout.write(self.style.MIGRATE_HEADING("="*60))
        
        # Get all Events from the specified test season
        test_events = Event.objects.filter(year=year).order_by('round')
        
        chaos_scores = []
        mae_scores = []
        event_names = []

        self.stdout.write(self.style.SQL_FIELD(
            f"{'Event Name':<30} | {'Chaos Score':>10} | {'MAE':>6}"
        ))
        self.stdout.write(self.style.SQL_FIELD("-" * 60))
        
        for event in test_events:
            chaos_score = self.calculate_chaos_score_for_event(event)
            mae = self.calculate_mae_for_event(event, model_name)
            
            if chaos_score is not None and mae is not None:
                chaos_scores.append(chaos_score)
                mae_scores.append(mae)
                event_names.append(event.name)
                
                self.stdout.write(self.style.SQL_COLTYPE(
                    f"{event.name:<30} | {chaos_score:>10.2f} | {mae:>6.2f}"
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f"{event.name:<30} | {'Insufficient Data':>22} | {'N/A':>6}"
                ))

        if not chaos_scores:
            self.stdout.write(self.style.ERROR("No valid data found for analysis."))
            return

        # --- Perform Linear Regression ---
        X = np.array(chaos_scores).reshape(-1, 1)
        y = np.array(mae_scores)

        model = LinearRegression()
        model.fit(X, y)

        r_sq = model.score(X, y)
        theoretical_mae_ceiling = model.intercept_  # This is the Y-value when Chaos Score = 0

        self.stdout.write(self.style.SUCCESS("="*60))
        self.stdout.write(self.style.SUCCESS("REGRESSION ANALYSIS RESULTS:"))
        self.stdout.write(self.style.SUCCESS(
            f"Coefficient of Determination (R²): {r_sq:.3f}"
        ))
        self.stdout.write(self.style.SUCCESS(
            f"Model Slope: {model.coef_[0]:.3f} (MAE increase per unit of Chaos)"
        ))
        self.stdout.write(self.style.SUCCESS(
            f"Theoretical Predictability Ceiling (MAE at Chaos=0): {theoretical_mae_ceiling:.2f}"
        ))

        # --- Create Publication-Quality Plot ---
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(chaos_scores, mae_scores, alpha=0.7, s=80)
        
        # Add race labels to points
        for i, name in enumerate(event_names):
            plt.annotate(name, (chaos_scores[i], mae_scores[i]), 
                         xytext=(5, 5), textcoords='offset points', 
                         fontsize=7, alpha=0.8)

        # Plot regression line
        x_range = np.linspace(min(chaos_scores)-0.5, max(chaos_scores)+0.5, 100).reshape(-1,1)
        y_pred = model.predict(x_range)
        plt.plot(x_range, y_pred, color='red', linestyle='--', 
                 label=f'MAE = {theoretical_mae_ceiling:.2f} + {model.coef_[0]:.2f}·Chaos')

        plt.xlabel('Calculated Chaos Score', fontsize=12)
        plt.ylabel('Prediction Error (MAE)', fontsize=12)
        plt.title(f'Impact of Race Chaos on Prediction Accuracy (F1 {year})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'f1_chaos_analysis_{year}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        self.stdout.write(self.style.SUCCESS(f"\nPlot saved as '{plot_filename}'."))