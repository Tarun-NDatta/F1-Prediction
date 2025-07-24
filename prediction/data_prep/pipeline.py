import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from django.db.models import F
from data.models import DriverPerformance, TeamPerformance, RaceResult
import logging

logger = logging.getLogger(__name__)

class F1DataPipeline:
    def __init__(self, test_size=0.2, random_state=42, impute_strategy='median'):
        self.test_size = test_size
        self.random_state = random_state
        self.impute_strategy = impute_strategy
        self.feature_names_ = None  # Store actual feature names used

        self.driver_features = [
            {'name': 'moving_avg_5', 'required': True},
            {'name': 'position_variance', 'required': True},
            {'name': 'qualifying_avg', 'required': True},
            {'name': 'circuit_affinity', 'required': False},
            {'name': 'quali_improvement', 'required': False},
            {'name': 'teammate_battle', 'required': False},
            {'name': 'wet_weather_perf', 'required': False},
            {'name': 'rivalry_performance', 'required': False},
            {'name': 'quali_race_delta', 'required': False},
            {'name': 'position_momentum', 'required': False},
            {'name': 'dnf_rate', 'required': True},
            {'name': 'pit_stop_avg', 'required': True},
            {'name': 'points_per_race', 'required': False},
        ]

        self.team_features = [
            {'name': 'dnf_rate', 'required': True},
            {'name': 'pit_stop_avg', 'required': True},
            {'name': 'reliability_score', 'required': False},
            {'name': 'development_slope', 'required': False},
            {'name': 'pit_stop_std', 'required': False},
            {'name': 'moving_avg_5', 'required': True},
            {'name': 'position_variance', 'required': True},
            {'name': 'qualifying_avg', 'required': True},
        ]

        self.all_features = (
            [f['name'] for f in self.driver_features] +
            [f['name'] for f in self.team_features]
        )

    def get_feature_names(self):
        if self.feature_names_ is None:
            # Return all possible feature names if not yet fitted
            return list(set(self.all_features))
        return self.feature_names_

    def get_required_features(self):
        return (
            [f['name'] for f in self.driver_features if f['required']] +
            [f['name'] for f in self.team_features if f['required']]
        )

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info("Loading race results...")
            race_query = (
                RaceResult.objects.filter(position__isnull=False)
                .select_related('session__event', 'driver', 'team')
                .annotate(
                    event_id=F('session__event_id'),
                    session_type=F('session__session_type__session_type'),
                    year=F('session__event__year'),
                    round_number=F('session__event__round'),  # Assumed field; adapt if different
                    circuit=F('session__event__circuit__name')  # Assumed field for circuit name
                )
            )

            race_results = pd.DataFrame.from_records(
                race_query.values(
                    'id', 'driver_id', 'team_id', 'position',
                    'event_id', 'session_type', 'grid_position',
                    'year', 'round_number', 'circuit'
                )
            )

            # Filter to race sessions only
            race_results = race_results[race_results['session_type'] == 'RACE']
            if race_results.empty:
                raise ValueError("No race results found after filtering")

            # Fix grid_position missing values
            race_results['grid_position'] = race_results['grid_position'].fillna(20).astype('int64')

            logger.info(f"Loaded {len(race_results)} race results")

            logger.info("Loading driver performance...")
            driver_fields = ['driver_id', 'event_id'] + [f['name'] for f in self.driver_features]
            driver_perf = pd.DataFrame.from_records(DriverPerformance.objects.all().values(*driver_fields))

            logger.info("Loading team performance...")
            team_fields = ['team_id', 'event_id'] + [f['name'] for f in self.team_features]
            team_perf = pd.DataFrame.from_records(TeamPerformance.objects.all().values(*team_fields))

            logger.info("Merging datasets...")
            df = pd.merge(
                race_results,
                driver_perf,
                on=['driver_id', 'event_id'],
                how='left',
                suffixes=('', '_driver')
            )

            df = pd.merge(
                df,
                team_perf,
                on=['team_id', 'event_id'],
                how='left',
                suffixes=('', '_team')
            )

            logger.info(f"Merged dataset contains {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise

    def _handle_missing_values(self, df):
        feature_variations = []
        for feature in self.get_feature_names():
            feature_variations.extend([
                feature,
                f"{feature}_driver",
                f"{feature}_team"
            ])

        existing_features = [f for f in feature_variations if f in df.columns]

        if not existing_features:
            raise ValueError("No features found in dataframe")

        missing = df[existing_features + ['position']].isnull().sum()
        if missing.any():
            logger.warning("Missing values found:\n" + missing[missing > 0].to_string())

            if self.impute_strategy == 'drop':
                original_size = len(df)
                required_cols = [f for f in self.get_required_features()
                                 if any(x in df.columns for x in (f, f"{f}_driver", f"{f}_team"))]
                df = df.dropna(subset=required_cols + ['position'])
                logger.info(f"Dropped {original_size - len(df)} records with missing values")
            else:
                if 'year' in df.columns and self.impute_strategy != 'drop':
                    years = df['year']
                    df = df.drop(columns=['year'])
                    df = self._impute_missing_values(df, existing_features)
                    df = pd.concat([df, years], axis=1)
                else:
                    df = self._impute_missing_values(df, existing_features)
        return df

    def _impute_missing_values(self, df, features):
        logger.info(f"Imputing missing values using {self.impute_strategy} strategy")
        y = df['position']
        X = df[features]

        if self.impute_strategy in ['mean', 'median']:
            impute_value = X.median() if self.impute_strategy == 'median' else X.mean()
            X = X.fillna(impute_value)
        elif self.impute_strategy == 'iterative':
            imputer = IterativeImputer(random_state=self.random_state)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        df = pd.concat([X, y], axis=1)
        return df

    def _validate_training_data(self, df):
        if len(df) == 0:
            raise ValueError("No data available after processing")

        if df['position'].isnull().any():
            raise ValueError("Target 'position' contains missing values after processing")

        if 'year' not in df.columns:
            raise ValueError("Year column missing from dataset")

        missing_required = []
        for feature in self.get_required_features():
            if not any(col in df.columns for col in (feature, f"{feature}_driver", f"{feature}_team")):
                missing_required.append(feature)

        if missing_required:
            raise ValueError(f"Missing required features: {missing_required}")

    def _select_features(self, df):
        selected_features = []
        for feature in list(set(self.all_features)):
            # Prefer team-specific for some features
            if feature in ['reliability_score', 'development_slope', 'pit_stop_std']:
                col = f"{feature}_team"
                if col in df.columns:
                    selected_features.append(col)
                    continue
            for suffix in ['', '_driver', '_team']:
                col = f"{feature}{suffix}"
                if col in df.columns:
                    selected_features.append(col)
                    break
        self.feature_names_ = selected_features
        return selected_features

    def _get_split_indices(self, df):
        bins = pd.qcut(df['position'], q=5, duplicates='drop')
        indices = np.arange(len(df))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=bins
        )
        return train_indices, test_indices

    def _split_data(self, df, return_dataframes=False):
        feature_columns = self._select_features(df)
        if not feature_columns:
            raise ValueError("No features available for training")

        train_indices, test_indices = self._get_split_indices(df)

        if return_dataframes:
            X = df[feature_columns]
            y = df['position']
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
        else:
            X = df[feature_columns].values
            y = df['position'].values
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

        logger.info(f"Data split complete - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def prepare_training_data(self, include_years=False, return_dataframes=False):
        try:
            df = self.load_data()
            if len(df) == 0:
                raise ValueError("No data available after loading")

            # Create lag features first to avoid missing columns in enhanced features
            df = self.create_lag_features(df)

            # Create enhanced features safely
            df = self.create_enhanced_features(df)

            df = self._handle_missing_values(df)
            self._validate_training_data(df)

            years = df['year'].values if 'year' in df.columns else None

            X_train, X_test, y_train, y_test = self._split_data(df, return_dataframes)

            if include_years and years is not None:
                train_indices, test_indices = self._get_split_indices(df)
                years_train = years[train_indices]
                years_test = years[test_indices]
                return X_train, X_test, y_train, y_test, years_train, years_test

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}", exc_info=True)
            raise

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for previous race positions per driver.
        Fills missing lag values with 20 (back of grid).
        """
        df = df.sort_values(['driver_id', 'year', 'round_number']).copy()

        df['position_last_race'] = df.groupby('driver_id')['position'].shift(1)
        df['position_2races_ago'] = df.groupby('driver_id')['position'].shift(2)
        df['position_3races_ago'] = df.groupby('driver_id')['position'].shift(3)

        df['position_last_race'] = df['position_last_race'].fillna(20)
        df['position_2races_ago'] = df['position_2races_ago'].fillna(20)
        df['position_3races_ago'] = df['position_3races_ago'].fillna(20)

        return df

    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced features using lag features and domain knowledge,
        safely handling missing columns.
        """
        # Weighted recent form - require lag features, fallback value if missing
        for lag_col in ['position_last_race', 'position_2races_ago', 'position_3races_ago']:
            if lag_col not in df.columns:
                logger.warning(f"Missing lag feature column '{lag_col}', filling with 20")
                df[lag_col] = 20

        df['weighted_form'] = (
            0.5 * df['position_last_race'] +
            0.3 * df['position_2races_ago'] +
            0.2 * df['position_3races_ago']
        )

        # Circuit affinity - average position per driver and circuit (rolling mean if desired)
        if {'driver_id', 'circuit', 'position'}.issubset(df.columns):
            df['circuit_affinity'] = df.groupby(['driver_id', 'circuit'])['position'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            df['circuit_affinity'] = df['circuit_affinity'].fillna(20)
        else:
            logger.warning("Missing one or more columns for circuit_affinity. Filling with 20.")
            df['circuit_affinity'] = 20

        # Tire delta if columns exist
        if 'soft_tire_laptime' in df.columns and 'medium_tire_laptime' in df.columns:
            df['tire_delta'] = df['soft_tire_laptime'] - df['medium_tire_laptime']
        else:
            logger.warning("Missing tire lap time columns, setting tire_delta = 0")
            df['tire_delta'] = 0

        # Teammate gap: qualifying time difference within same team, year, round
        if {'team_id', 'year', 'round_number', 'qualifying_time'}.issubset(df.columns):
            def calc_teammate_gap(group):
                mean_q = group['qualifying_time'].mean()
                return group['qualifying_time'] - mean_q

            df['teammate_gap'] = df.groupby(['team_id', 'year', 'round_number']).apply(calc_teammate_gap).reset_index(level=[0,1,2], drop=True)
        else:
            logger.warning("Missing columns for teammate_gap, setting to zero")
            df['teammate_gap'] = 0

        return df

    def get_preprocessing_pipeline(self):
        steps = []
        if self.impute_strategy == 'iterative':
            steps.append(('imputer', IterativeImputer(random_state=self.random_state)))
        else:
            # default is median imputation
            steps.append(('imputer', SimpleImputer(strategy='median')))
        steps.append(('scaler', StandardScaler()))
        return Pipeline(steps)