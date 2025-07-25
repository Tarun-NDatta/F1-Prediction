import os
from django.core.management.base import BaseCommand
from data.models import Event, TrackSpecialization
from .enhanced_pipeline import EnhancedF1Pipeline
import traceback
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train and use CatBoost model with track specialization and existing ensemble predictions'

    def add_arguments(self, parser):
        parser.add_argument('--mode', type=str, required=True,
                        choices=['initialize', 'train', 'predict', 'evaluate', 'analyze', 'generate'],
                        help='Mode of operation')
        parser.add_argument('--year', type=int,
                            help='Year for prediction or evaluation')
        parser.add_argument('--round', type=int,
                            help='Round number for prediction or evaluation')
        parser.add_argument('--model-dir', type=str,
                            default=r"C:\Users\tarun\diss\td188",
                            help='Directory containing trained models')
        parser.add_argument('--use-openf1', action='store_true',
                            help='Use OpenF1 live data for predictions (future feature)')
        parser.add_argument('--save-model', action='store_true',
                            help='Save trained CatBoost model')
        

    def handle(self, *args, **options):
        try:
            pipeline = EnhancedF1Pipeline(model_dir=options['model_dir'])
            
            if options['mode'] == 'initialize':
                self.initialize_track_data()
            elif options['mode'] == 'train':
                self.train_catboost_model(pipeline, options['save_model'])
            elif options['mode'] == 'predict':
                if not options['year'] or not options['round']:
                    self.stdout.write(self.style.ERROR("Year and round required for prediction"))
                    return
                self.make_predictions(pipeline, options['year'], options['round'], options['use_openf1'])
            elif options['mode'] == 'evaluate':
                if not options['year'] or not options['round']:
                    self.stdout.write(self.style.ERROR("Year and round required for evaluation"))
                    return
                self.evaluate_predictions(pipeline, options['year'], options['round'])
            elif options['mode'] == 'analyze':
                self.analyze_track_performance(pipeline)
            elif options['mode'] == 'generate':
                self.generate_base_predictions(pipeline)

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
            logger.error(f"Command error: {str(e)}", exc_info=True)
            traceback.print_exc()
    def generate_base_predictions(self, pipeline):
        """Generate base model predictions for historical races"""
        self.stdout.write("Generating base model predictions for 2022-2024...")
        try:
            saved_count = pipeline.generate_base_predictions(start_year=2024, end_year=2024)  # Start with 2024
            self.stdout.write(self.style.SUCCESS(f"Generated {saved_count} predictions"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error generating predictions: {str(e)}"))
            logger.error(f"Generate predictions error: {str(e)}", exc_info=True)
    
    def initialize_track_data(self):
        """Initialize track specialization data"""
        self.stdout.write("Initializing track specialization data...")
        
        try:
            created, updated = TrackSpecialization.initialize_track_data()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Track initialization complete: {created} created, {updated} updated"
                )
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error initializing track data: {str(e)}"))
            logger.error(f"Track initialization error: {str(e)}", exc_info=True)

    def train_catboost_model(self, pipeline, save_model=False):
        """Train the CatBoost model"""
        self.stdout.write("Training CatBoost model with track specialization...")
        
        try:
            # Load existing models first
            if not pipeline.load_existing_models():
                self.stdout.write(self.style.WARNING("Could not load existing models, continuing with database predictions"))
            
            # Prepare training data
            self.stdout.write("Preparing training data...")
            training_df = pipeline.prepare_catboost_training_data()
            
            if len(training_df) == 0:
                self.stdout.write(self.style.ERROR("No training data available"))
                return
            
            self.stdout.write(f"Training on {len(training_df)} samples")
            
            # Train model
            model, importance_df = pipeline.train_catboost_model(training_df)
            
            # Save model if requested
            if save_model:
                model_path = pipeline.save_catboost_model()
                self.stdout.write(self.style.SUCCESS(f"Model saved to {model_path}"))
            
            # Display feature importance
            self.stdout.write("\n=== Top 10 Feature Importance ===")
            for _, row in importance_df.head(10).iterrows():
                self.stdout.write(f"{row['feature']:<25} {row['importance']:.3f}")
            
            self.stdout.write(self.style.SUCCESS("CatBoost training completed successfully!"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Training error: {str(e)}"))
            logger.error(f"Training error: {str(e)}", exc_info=True)
            raise

    def make_predictions(self, pipeline, year, round_num, use_openf1=False):
        """Make predictions for a specific race"""
        try:
            event = Event.objects.get(year=year, round=round_num)
            self.stdout.write(f"Making predictions for {event.name} ({year} Round {round_num})")
            
            # Load models
            if not pipeline.load_existing_models():
                self.stdout.write(self.style.WARNING("Could not load existing ensemble models"))
            
            if not pipeline.load_catboost_model():
                self.stdout.write(self.style.ERROR("Could not load CatBoost model. Please train first."))
                return
            
            # Make predictions
            predictions_df = pipeline.predict_race(year, round_num, use_openf1)
            
            # Display predictions
            self.display_predictions(predictions_df, event)
            
            # Save to database
            saved_count = pipeline.save_predictions_to_db(predictions_df, event, use_openf1)
            self.stdout.write(self.style.SUCCESS(f"Saved {saved_count} predictions to database"))
            
        except Event.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Event not found: {year} Round {round_num}"))
            logger.error(f"Event not found: {year} Round {round_num}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Prediction error: {str(e)}"))
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise

    def display_predictions(self, predictions_df, event):
        """Display race predictions in a formatted table"""
        self.stdout.write(f"\n=== CatBoost Predictions for {event.name} ===")
        self.stdout.write(f"Track Category: {predictions_df.iloc[0]['track_category']}")
        self.stdout.write("")
        
        # Header
        self.stdout.write(f"{'Pos':<4} {'Driver':<20} {'CatBoost':<9} {'Ensemble':<9} {'Ridge':<8} {'XGBoost':<8} {'Qual':<5}")
        self.stdout.write("-" * 65)
        
        # Predictions
        for _, row in predictions_df.iterrows():
            self.stdout.write(
                f"{int(row['final_predicted_position']):<4} "
                f"{row['driver']:<20} "
                f"{row['catboost_prediction']:<9.2f} "
                f"{row['ensemble_prediction']:<9.2f} "
                f"{row['ridge_prediction']:<8.2f} "
                f"{row['xgboost_prediction']:<8.2f} "
                f"{int(row['qualifying_position']):<5}"
            )
        
        # Track characteristics
        self.stdout.write(f"\n=== Track Characteristics ===")
        self.stdout.write(f"Power Sensitivity: {predictions_df.iloc[0]['track_power_sensitivity']:.1f}/10")
        self.stdout.write(f"Overtaking Difficulty: {predictions_df.iloc[0]['track_overtaking_difficulty']:.1f}/10")
        self.stdout.write(f"Qualifying Importance: {predictions_df.iloc[0]['track_qualifying_importance']:.1f}/10")

    def evaluate_predictions(self, pipeline, year, round_num):
        """Evaluate predictions against actual results"""
        try:
            event = Event.objects.get(year=year, round=round_num)
            self.stdout.write(f"Evaluating predictions for {event.name} ({year} Round {round_num})")
            
            # Load CatBoost model
            if not pipeline.load_catboost_model():
                self.stdout.write(self.style.ERROR("Could not load CatBoost model"))
                return
            
            # Compare with actual results
            comparison_result = pipeline.compare_with_actual_results(event)
            
            if comparison_result is None:
                self.stdout.write(self.style.WARNING("No data available for comparison"))
                return
            
            comparison_df, metrics = comparison_result
            
            # Display comparison
            self.stdout.write(f"\n=== Prediction vs Actual Results ===")
            self.stdout.write(f"{'Driver':<20} {'Actual':<7} {'CatBoost':<9} {'Ensemble':<9} {'Diff CB':<8} {'Diff Ens':<8}")
            self.stdout.write("-" * 70)
            
            for _, row in comparison_df.iterrows():
                cb_diff = row['catboost_prediction'] - row['actual_position']
                ens_diff = row['ensemble_prediction'] - row['actual_position']
                
                self.stdout.write(
                    f"{row['driver']:<20} "
                    f"{int(row['actual_position']):<7} "
                    f"{row['catboost_prediction']:<9.2f} "
                    f"{row['ensemble_prediction']:<9.2f} "
                    f"{cb_diff:+8.2f} "
                    f"{ens_diff:+8.2f}"
                )
            
            # Display metrics
            self.stdout.write(f"\n=== Performance Metrics ===")
            self.stdout.write(f"{'Metric':<20} {'CatBoost':<10} {'Ensemble':<10} {'Improvement':<12}")
            self.stdout.write("-" * 55)
            self.stdout.write(f"{'MAE':<20} {metrics['catboost_mae']:<10.3f} {metrics['ensemble_mae']:<10.3f} {metrics['ensemble_mae'] - metrics['catboost_mae']:+12.3f}")
            self.stdout.write(f"{'RMSE':<20} {metrics['catboost_rmse']:<10.3f} {metrics['ensemble_rmse']:<10.3f} {metrics['ensemble_rmse'] - metrics['catboost_rmse']:+12.3f}")
            self.stdout.write(f"{'R²':<20} {metrics['catboost_r2']:<10.3f} {metrics['ensemble_r2']:<10.3f} {metrics['catboost_r2'] - metrics['ensemble_r2']:+12.3f}")
            self.stdout.write(f"{'Spearman':<20} {metrics['catboost_spearman']:<10.3f} {metrics['ensemble_spearman']:<10.3f} {metrics['catboost_spearman'] - metrics['ensemble_spearman']:+12.3f}")
            
            # Determine which model performed better
            if metrics['catboost_mae'] < metrics['ensemble_mae']:
                self.stdout.write(self.style.SUCCESS(f"\nCatBoost outperformed ensemble by {metrics['ensemble_mae'] - metrics['catboost_mae']:.3f} MAE"))
            else:
                self.stdout.write(self.style.WARNING(f"\nEnsemble outperformed CatBoost by {metrics['catboost_mae'] - metrics['ensemble_mae']:.3f} MAE"))
            
        except Event.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Event not found: {year} Round {round_num}"))
            logger.error(f"Event not found: {year} Round {round_num}")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Evaluation error: {str(e)}"))
            logger.error(f"Evaluation error: {str(e)}", exc_info=True)
            raise

    def analyze_track_performance(self, pipeline):
        """Analyze model performance by track category"""
        try:
            self.stdout.write("Analyzing performance by track category...")
            
            analysis_df = pipeline.get_track_performance_analysis()
            
            if analysis_df is None:
                self.stdout.write(self.style.WARNING("No data available for track analysis"))
                return
            
            self.stdout.write(f"\n=== Track Category Performance Analysis ===")
            self.stdout.write(f"{'Category':<15} {'Samples':<8} {'CatBoost MAE':<12} {'Ensemble MAE':<12} {'Improvement':<12}")
            self.stdout.write("-" * 65)
            
            for _, row in analysis_df.iterrows():
                improvement_color = self.style.SUCCESS if row['improvement'] > 0 else self.style.ERROR
                
                self.stdout.write(
                    f"{row['track_category']:<15} "
                    f"{row['sample_count']:<8} "
                    f"{row['catboost_mae']:<12.3f} "
                    f"{row['ensemble_mae']:<12.3f} "
                    f"{improvement_color(f'{row['improvement']:+12.3f}')}"
                )
            
            # Summary
            total_improvement = analysis_df['improvement'].mean()
            if total_improvement > 0:
                self.stdout.write(self.style.SUCCESS(f"\nOverall: CatBoost shows {total_improvement:.3f} average MAE improvement"))
            else:
                self.stdout.write(self.style.WARNING(f"\nOverall: Ensemble shows {abs(total_improvement):.3f} average MAE advantage"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Analysis error: {str(e)}"))
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            raise