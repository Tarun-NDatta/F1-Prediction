import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from django.db.models import F
from data.models import DriverPerformance, TeamPerformance, RaceResult, Session
import logging

logger = logging.getLogger(__name__)

class F1DataPipeline:
    def __init__(self, test_size=0.2, random_state=42, impute_strategy='median'):
        self.test_size = test_size
        self.random_state = random_state
        self.impute_strategy = impute_strategy
        self.feature_names_ = None  # Store actual feature names used
        
        # Define driver-specific and team-specific features
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
            {'name': 'points_per_race', 'required': False}
        ]
        
        self.team_features = [
            {'name': 'dnf_rate', 'required': True},
            {'name': 'pit_stop_avg', 'required': True},
            {'name': 'reliability_score', 'required': False},
            {'name': 'development_slope', 'required': False},
            {'name': 'pit_stop_std', 'required': False},
            {'name': 'moving_avg_5', 'required': True},
            {'name': 'position_variance', 'required': True},
            {'name': 'qualifying_avg', 'required': True}
        ]
        
        # Combined list of all feature names
        self.all_features = (
            [f['name'] for f in self.driver_features] + 
            [f['name'] for f in self.team_features]
        )
    
    def get_feature_names(self):
        """Return feature names used in the last data preparation"""
        if self.feature_names_ is None:
            # Return potential feature names if not yet fitted
            return list(set(self.all_features))
        return self.feature_names_
    
    def get_required_features(self):
        return (
            [f['name'] for f in self.driver_features if f['required']] +
            [f['name'] for f in self.team_features if f['required']]
        )

    def load_data(self):
        """Load data with year information"""
        try:
            logger.info("Loading race results...")
            race_query = (
                RaceResult.objects.filter(position__isnull=False)
                .select_related('session__event', 'driver', 'team')
                .annotate(
                    event_id=F('session__event_id'),
                    session_type=F('session__session_type__session_type'),
                    year=F('session__event__year')
                )
            )
            
            # Add year to the results
            race_results = pd.DataFrame.from_records(
                race_query.values(
                    'id', 'driver_id', 'team_id', 'position',
                    'event_id', 'session_type', 'grid_position', 'year'
                )
            )
            
            race_results = race_results[race_results['session_type'] == 'RACE']
            
            if len(race_results) == 0:
                raise ValueError("No race results found after filtering")
            
            logger.info(f"Loaded {len(race_results)} race results")

            # Load driver performance with driver-specific fields
            logger.info("Loading driver performance...")
            driver_fields = ['driver_id', 'event_id'] + [f['name'] for f in self.driver_features]
            driver_perf = pd.DataFrame.from_records(
                DriverPerformance.objects.all().values(*driver_fields)
            )
            
            # Load team performance with team-specific fields
            logger.info("Loading team performance...")
            team_fields = ['team_id', 'event_id'] + [f['name'] for f in self.team_features]
            team_perf = pd.DataFrame.from_records(
                TeamPerformance.objects.all().values(*team_fields)
            )
            
            # Handle missing grid positions - fix pandas warning
            race_results['grid_position'] = race_results['grid_position'].fillna(20).astype('int64')
            #race_results['grid_position'] = race_results['grid_position'].astype('int64')
            
            # Merge data with suffix handling
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
        """Handle missing values while preserving year"""
        # Get all possible feature variations
        feature_variations = []
        for feature in self.get_feature_names():
            feature_variations.extend([
                feature,
                f"{feature}_driver",
                f"{feature}_team"
            ])
        
        # Find which features actually exist in the dataframe
        existing_features = [f for f in feature_variations if f in df.columns]
        
        if not existing_features:
            raise ValueError("No features found in dataframe")
            
        missing = df[existing_features + ['position']].isnull().sum()
        
        if missing.any():
            logger.warning("Missing values found:\n" + missing[missing > 0].to_string())
            
            if self.impute_strategy == 'drop':
                original_size = len(df)
                required_cols = [f for f in self.get_required_features() 
                               if f in df.columns or 
                               f"{f}_driver" in df.columns or 
                               f"{f}_team" in df.columns]
                df = df.dropna(subset=required_cols + ['position'])
                logger.info(f"Dropped {original_size - len(df)} records with missing values")
            else:
                # Preserve year column during imputation
                if 'year' in df.columns and self.impute_strategy != 'drop':
                    years = df['year']
                    df = df.drop(columns=['year'])
                    
                    # Perform imputation
                    df = self._impute_missing_values(df, existing_features)
                    
                    # Add year back after imputation
                    df = pd.concat([df, years], axis=1)
                else:
                    df = self._impute_missing_values(df, existing_features)
        
        return df

    def _impute_missing_values(self, df, features):
        """Impute missing values using selected strategy"""
        logger.info(f"Imputing missing values using {self.impute_strategy} strategy")
        
        # Separate features and target
        y = df['position']
        X = df[features]
        
        if self.impute_strategy in ['mean', 'median']:
            impute_value = X.median() if self.impute_strategy == 'median' else X.mean()
            X = X.fillna(impute_value)
        elif self.impute_strategy == 'iterative':
            imputer = IterativeImputer(random_state=self.random_state)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Recombine
        df = pd.concat([X, y], axis=1)
        return df

    def _validate_training_data(self, df):
        """Validate data including year information"""
        if len(df) == 0:
            raise ValueError("No data available after processing")
        
        # Check target variable
        if df['position'].isnull().any():
            raise ValueError("Position still contains missing values after processing")
        
        # Check year exists
        if 'year' not in df.columns:
            raise ValueError("Year column missing from dataset")
        
        # Check required features
        missing_required = []
        for feature in self.get_required_features():
            if (feature not in df.columns and 
                f"{feature}_driver" not in df.columns and 
                f"{feature}_team" not in df.columns):
                missing_required.append(feature)
        
        if missing_required:
            raise ValueError(f"Missing required features: {missing_required}")

    def _select_features(self, df):
        """Select the best available version of each feature"""
        selected_features = []
        for feature in list(set(self.all_features)):
            # Handle team-specific features
            if feature in ['reliability_score', 'development_slope', 'pit_stop_std']:
                col = f"{feature}_team"  # Always use team version
                if col in df.columns:
                    selected_features.append(col)
                    continue
                    
            # For other features, prefer driver-specific, then team, then base
            for suffix in ['', '_driver', '_team']:
                col = f"{feature}{suffix}"
                if col in df.columns:
                    selected_features.append(col)
                    break
        
        # Store the actual feature names used
        self.feature_names_ = selected_features
        return selected_features

    def _get_split_indices(self, df):
        """Get indices for time-based split"""
        # Create stratified bins based on position distribution
        bins = pd.qcut(df['position'], q=5, duplicates='drop')
        
        # Get indices for train/test split
        indices = np.arange(len(df))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=bins
        )
        return train_indices, test_indices

    def _split_data(self, df, return_dataframes=False):
        """Split data into features and target with option to return DataFrames"""
        feature_columns = self._select_features(df)
        
        if not feature_columns:
            raise ValueError("No features available for training")
        
        # Get indices for consistent splitting
        train_indices, test_indices = self._get_split_indices(df)
        
        if return_dataframes:
            # Return DataFrames (useful for getting column names)
            X = df[feature_columns]
            y = df['position']
            
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
        else:
            # Return numpy arrays (for Ridge regression compatibility)
            X = df[feature_columns].values
            y = df['position'].values
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
        
        logger.info(
            f"Data split complete - Train: {len(X_train)}, Test: {len(X_test)}"
        )
        return X_train, X_test, y_train, y_test

    def prepare_training_data(self, include_years=False, return_dataframes=False):
        """
        Prepare training data with options for different output formats
        
        Args:
            include_years: If True, also return year information
            return_dataframes: If True, return DataFrames instead of numpy arrays
        """
        try:
            df = self.load_data()
            
            if len(df) == 0:
                raise ValueError("No data available after loading")
            
            df = self._handle_missing_values(df)
            self._validate_training_data(df)
            
            # Extract year information before splitting
            years = df['year'].values if 'year' in df.columns else None
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(df, return_dataframes)
            
            if include_years and years is not None:
                # Get indices for train/test split
                train_indices, test_indices = self._get_split_indices(df)
                years_train = years[train_indices]
                years_test = years[test_indices]
                return X_train, X_test, y_train, y_test, years_train, years_test
            
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}", exc_info=True)
            raise

    def create_enhanced_features(self, df):
        """Add these to your existing feature pipeline"""
        # 1. Recent form (weighted average)
        df['weighted_form'] = (
            0.5 * df['position_last_race'] + 
            0.3 * df['position_2races_ago'] + 
            0.2 * df['position_3races_ago']
        )
        
        # 2. Circuit affinity (historical performance at this track)
        df['circuit_affinity'] = df.groupby(['driver_id', 'circuit'])['position'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        
        # 3. Tire performance delta
        df['tire_delta'] = df['soft_tire_laptime'] - df['medium_tire_laptime']
        
        # 4. Head-to-head teammate comparison
        df['teammate_gap'] = df.groupby(['team', 'year', 'round']).apply(
            lambda g: g['qualifying_time'] - g['qualifying_time'].mean()
        ).reset_index(level=[0,1,2], drop=True)
        
        return df
    
    def get_preprocessing_pipeline(self):
        """Get preprocessing pipeline with optional imputation"""
        steps = []
        
        if self.impute_strategy == 'iterative':
            steps.append(('imputer', IterativeImputer(random_state=self.random_state)))
        
        steps.append(('scaler', StandardScaler()))
        
        return Pipeline(steps)