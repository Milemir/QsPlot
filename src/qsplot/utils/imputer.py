"""
FastImputer: A high-performance, Scikit-Learn compatible imputer for financial time-series data.
Handles missing data using various strategies including interpolation, forward/backward fill, and statistical measures.
"""

from typing import List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class FastImputer(BaseEstimator, TransformerMixin):
    """
    High-performance imputer for financial time-series data.

    Supported strategies:
        - 'linear': Linear interpolation.
        - 'time': Time-weighted interpolation (requires DatetimeIndex).
        - 'ffill': Forward fill.
        - 'bfill': Backward fill.
        - 'mean': Fill with column mean.
        - 'median': Fill with column median.
        - 'zero': Fill with 0.

    Attributes:
        strategy (str): The imputation strategy.
        columns (List[str] | None): Columns to apply imputation to. If None, applies to all numeric columns.
        statistics_ (pd.Series): Computed statistics (mean/median) for 'mean'/'median' strategies.
    """

    def __init__(self, strategy: str = 'linear', columns: Optional[List[str]] = None):
        """
        Initialize the FastImputer.

        Args:
            strategy: The imputation strategy to use. Defaults to 'linear'.
            columns: List of column names to impute. If None, all numeric columns are used.
        """
        self.strategy = strategy
        self.columns = columns
        self.statistics_ = None

    def fit(self, X: pd.DataFrame, y=None) -> 'FastImputer':
        """
        Fit the imputer on X.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self: Returns the instance itself.
        """
        # Validate strategy
        valid_strategies = {'linear', 'time', 'ffill', 'bfill', 'mean', 'median', 'zero'}
        if self.strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Supported: {valid_strategies}")

        # Determine columns to process if not explicitly provided
        if self.columns is None:
            # Select only numeric columns to avoid errors on strings/objects
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Verify user-provided columns exist
            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        # Compute statistics for statistical strategies
        if self.strategy == 'mean':
            self.statistics_ = X[self.columns].mean()
        elif self.strategy == 'median':
            self.statistics_ = X[self.columns].median()
        else:
            self.statistics_ = pd.Series(0, index=self.columns) # Placeholder/Not used for other strategies

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute all missing values in X.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with missing values filled.
        """
        check_is_fitted(self, 'statistics_')

        X_transformed = X.copy()
        target_cols = self.columns

        if self.strategy == 'linear':
            X_transformed[target_cols] = X_transformed[target_cols].interpolate(method='linear', limit_direction='both')

        elif self.strategy == 'time':
            if not isinstance(X.index, pd.DatetimeIndex):
                warnings.warn("Strategy 'time' requires DatetimeIndex. Falling back to 'linear'.", UserWarning)
                X_transformed[target_cols] = X_transformed[target_cols].interpolate(method='linear', limit_direction='both')
            else:
                X_transformed[target_cols] = X_transformed[target_cols].interpolate(method='time', limit_direction='both')

        elif self.strategy == 'ffill':
            X_transformed[target_cols] = X_transformed[target_cols].ffill()

        elif self.strategy == 'bfill':
            X_transformed[target_cols] = X_transformed[target_cols].bfill()

        elif self.strategy == 'mean':
            X_transformed[target_cols] = X_transformed[target_cols].fillna(self.statistics_)

        elif self.strategy == 'median':
            X_transformed[target_cols] = X_transformed[target_cols].fillna(self.statistics_)

        elif self.strategy == 'zero':
            X_transformed[target_cols] = X_transformed[target_cols].fillna(0)

        return X_transformed


if __name__ == "__main__":
    print("Running FastImputer Demo...")

    # Create a sample financial DataFrame with NaNs
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = {
        'Price': [100.0, 102.0, np.nan, 104.0, np.nan, 105.0, 110.0, np.nan, 108.0, 109.0],
        'Volume': [1000, np.nan, 1200, 1500, np.nan, 1300, np.nan, 1600, 1650, 1700],
        'Symbol': ['AAPL'] * 10  # Non-numeric column to ensure preservation
    }
    df = pd.DataFrame(data, index=dates)
    
    # Introduce an irregularity to test 'time' interpolation
    df_irregular = df.drop(df.index[[1, 5]]) 

    print("\nOriginal DataFrame (with NaNs):")
    print(df_irregular)

    # 1. Linear Interpolation
    print("\n--- Strategy: 'linear' ---")
    imputer_linear = FastImputer(strategy='linear')
    df_linear = imputer_linear.fit_transform(df_irregular)
    print(df_linear)

    # 2. Time Interpolation
    print("\n--- Strategy: 'time' ---")
    imputer_time = FastImputer(strategy='time')
    df_time = imputer_time.fit_transform(df_irregular)
    print(df_time)

    # 3. Forward Fill
    print("\n--- Strategy: 'ffill' ---")
    imputer_ffill = FastImputer(strategy='ffill')
    df_ffill = imputer_ffill.fit_transform(df_irregular)
    print(df_ffill)
    
    # 4. Fill with Median (Specific Column)
    print("\n--- Strategy: 'median' (Price only) ---")
    imputer_median = FastImputer(strategy='median', columns=['Price'])
    df_median = imputer_median.fit_transform(df_irregular)
    print(df_median)

    # 5. Non-Datetime Index Error Handling Check
    print("\n--- Error Handling Check (Non-Datetime Index with 'time' strategy) ---")
    df_no_date = df_irregular.reset_index(drop=True)
    imputer_err = FastImputer(strategy='time')
    # This should warn and fallback to linear
    df_fallback = imputer_err.fit_transform(df_no_date)
    print("Fallback successful (See warning above).")
