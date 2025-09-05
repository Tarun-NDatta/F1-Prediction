import os
from django.core.management.base import BaseCommand
from django.conf import settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive 'Agg' backend to avoid font logging
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


from prediction.analysis.chaos import (
    analyze_prediction_errors_by_events,
    CounterfactualAnalyzer,
    RaceEventClassifier,
)


class Command(BaseCommand):
    help = 'Analyze chaos impact on ML prediction accuracy and generate CSVs/figures.'

    def add_arguments(self, parser):
        parser.add_argument('--season', type=int, default=None, help='Season year (e.g., 2025). If omitted, analyzes all years.')
        parser.add_argument('--start', type=int, default=2022, help='Start year inclusive when season omitted')
        parser.add_argument('--end', type=int, default=2025, help='End year inclusive when season omitted')
        parser.add_argument('--model', type=str, default='catboost_ensemble')
        parser.add_argument('--outdir', type=str, default='chaos_analysis')

    def handle(self, *args, **options):
        season = options['season']
        start = options['start']
        end = options['end']
        model = options['model']
        outdir = os.path.join(settings.BASE_DIR, options['outdir'])
        os.makedirs(outdir, exist_ok=True)

        if season is None:
            # Combine multiple seasons
            dfs = []
            for yr in range(start, end + 1):
                _df = analyze_prediction_errors_by_events(season=yr, model_name=model)
                if not _df.empty:
                    dfs.append(_df.assign(year=yr))
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            df = analyze_prediction_errors_by_events(season=season, model_name=model)
            if not df.empty:
                df['year'] = season

        if df.empty:
            self.stdout.write(self.style.WARNING('No predictions found for analysis.'))
            return

        # Save raw analysis
        csv_path = os.path.join(outdir, f'chaos_errors_{season}_{model}.csv')
        df.to_csv(csv_path, index=False)

        cf = CounterfactualAnalyzer(df)
        clean_mae = cf.analyze_clean_race_accuracy()
        perfect_chaos_mae = cf.simulate_perfect_chaos_prediction()
        grouped = cf.group_mae()

        # Bootstrap CIs
        mask_clean_unaffected = (df['race_category'] == 'clean') & (~df['driver_affected'])
        mask_chaotic = (df['race_category'] == 'chaotic')
        mae_clean, l_clean, u_clean = cf.bootstrap_ci(mask_clean_unaffected)
        mae_chaotic, l_ch, u_ch = cf.bootstrap_ci(mask_chaotic)

        # Significance tests
        sig = cf.significance_test(mask_clean_unaffected, mask_chaotic)

        # Summary table
        summary = {
            'overall_mae': float(np.mean(df['error'])),
            'clean_unaffected_mae': float(clean_mae),
            'perfect_chaos_mae': float(perfect_chaos_mae),
            'ci_clean_low': l_clean,
            'ci_clean_high': u_clean,
            'ci_chaotic_low': l_ch,
            'ci_chaotic_high': u_ch,
            'sig_method': sig.get('method'),
            'sig_p_value': sig.get('p_value'),
        }
        pd.DataFrame([summary]).to_csv(os.path.join(outdir, f'summary_{season}_{model}.csv'), index=False)
        grouped.to_csv(os.path.join(outdir, f'group_mae_{season}_{model}.csv'), index=False)

        # Visualizations
        plt.style.use('seaborn-v0_8')
        tag = f"{season or f'{start}-{end}'}_{model}"

        # Box plot: MAE by race category
        fig, ax = plt.subplots(figsize=(8, 5))
        df.boxplot(column='error', by='race_category', ax=ax)
        ax.set_title('Prediction Error by Race Chaos Category')
        ax.set_ylabel('Absolute Error (positions)')
        fig.suptitle('')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'box_mae_by_category_{tag}.png'), dpi=200)
        plt.close(fig)

        # Histogram: affected vs unaffected
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[df['driver_affected']]['error'], bins=20, alpha=0.6, label='Affected')
        ax.hist(df[~df['driver_affected']]['error'], bins=20, alpha=0.6, label='Unaffected')
        ax.set_title('Error Distribution: Affected vs Unaffected Drivers')
        ax.set_xlabel('Absolute Error (positions)')
        ax.set_ylabel('Frequency')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'hist_affected_vs_unaffected_{tag}.png'), dpi=200)
        plt.close(fig)

        # Scatter: chaos score vs error
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df['chaos_score'], df['error'], alpha=0.7)
        ax.set_title('Chaos Score vs Prediction Error')
        ax.set_xlabel('Race Chaos Score (0-10)')
        ax.set_ylabel('Absolute Error (positions)')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'scatter_chaos_vs_error_{tag}.png'), dpi=200)
        plt.close(fig)

        # Targeted analysis by position groups (front, midfield, back)
        def pos_group(pos):
            if pos <= 6: return 'front'
            if pos <= 15: return 'midfield'
            return 'back'
        df['pos_group'] = df['actual_pos'].apply(pos_group)
        grp_pg = df.groupby(['pos_group','race_category'])['error'].mean().reset_index()
        grp_pg.to_csv(os.path.join(outdir, f'position_group_mae_{tag}.csv'), index=False)

        fig, ax = plt.subplots(figsize=(8,5))
        for pg in ['front','midfield','back']:
            sub = grp_pg[grp_pg['pos_group']==pg]
            ax.plot(sub['race_category'], sub['error'], marker='o', label=pg)
        ax.set_title('MAE by Chaos Category and Position Group')
        ax.set_ylabel('MAE')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'position_group_mae_{tag}.png'), dpi=200)
        plt.close(fig)

        # Print narrative summary to console
        self.stdout.write('--- Chaos Impact Summary ---')
        self.stdout.write(f"Season: {season} | Model: {model}")
        self.stdout.write(f"Overall MAE: {summary['overall_mae']:.2f}")
        self.stdout.write(f"Clean (unaffected) MAE: {summary['clean_unaffected_mae']:.2f} (95% CI {l_clean:.2f}-{u_clean:.2f})")
        self.stdout.write(f"Perfect chaos knowledge MAE: {summary['perfect_chaos_mae']:.2f}")
        self.stdout.write(f"Significance ({summary['sig_method']}): p={summary['sig_p_value']:.4f}")
        self.stdout.write(self.style.SUCCESS(f"Artifacts saved to: {outdir}"))

