"""
Tests for FastImputer - Scikit-learn compatible imputer for financial time-series data.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from qsplot.utils.imputer import FastImputer


class TestFastImputerStrategies:
    """Test all imputation strategies."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with NaN values."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        return pd.DataFrame({
            "Price": [100.0, 102.0, np.nan, 104.0, np.nan, 105.0, 110.0, np.nan, 108.0, 109.0],
            "Volume": [1000, np.nan, 1200, 1500, np.nan, 1300, np.nan, 1600, 1650, 1700],
            "Symbol": ["AAPL"] * 10
        }, index=dates)

    def test_linear_interpolation(self, sample_df):
        """Test linear interpolation fills NaN values."""
        imputer = FastImputer(strategy='linear')
        result = imputer.fit_transform(sample_df)
        
        # Check no NaN in numeric columns
        assert not result['Price'].isna().any()
        assert not result['Volume'].isna().any()
        # Non-numeric column should be untouched
        assert (result['Symbol'] == 'AAPL').all()

    def test_time_interpolation_with_datetime_index(self, sample_df):
        """Test time-weighted interpolation with DatetimeIndex."""
        imputer = FastImputer(strategy='time')
        result = imputer.fit_transform(sample_df)
        
        assert not result['Price'].isna().any()
        assert not result['Volume'].isna().any()

    def test_time_interpolation_fallback_without_datetime(self, sample_df):
        """Test fallback to linear when index is not DatetimeIndex."""
        df_no_date = sample_df.reset_index(drop=True)
        imputer = FastImputer(strategy='time')
        
        with pytest.warns(UserWarning, match="requires DatetimeIndex"):
            result = imputer.fit_transform(df_no_date)
        
        # Should still fill values (fell back to linear)
        assert not result['Price'].isna().any()

    def test_ffill_strategy(self, sample_df):
        """Test forward fill strategy."""
        imputer = FastImputer(strategy='ffill')
        result = imputer.fit_transform(sample_df)
        
        # Price at index 2 (NaN) should be filled with 102.0 (previous value)
        assert result['Price'].iloc[2] == 102.0

    def test_bfill_strategy(self, sample_df):
        """Test backward fill strategy."""
        imputer = FastImputer(strategy='bfill')
        result = imputer.fit_transform(sample_df)
        
        # Price at index 2 (NaN) should be filled with 104.0 (next value)
        assert result['Price'].iloc[2] == 104.0

    def test_mean_strategy(self, sample_df):
        """Test mean fill strategy."""
        imputer = FastImputer(strategy='mean')
        result = imputer.fit_transform(sample_df)
        
        # Calculate expected mean (excluding NaN)
        expected_mean = sample_df['Price'].mean()
        assert not result['Price'].isna().any()
        # All NaN positions should be filled with mean
        assert result['Price'].iloc[2] == pytest.approx(expected_mean)

    def test_median_strategy(self, sample_df):
        """Test median fill strategy."""
        imputer = FastImputer(strategy='median')
        result = imputer.fit_transform(sample_df)
        
        expected_median = sample_df['Price'].median()
        assert result['Price'].iloc[2] == pytest.approx(expected_median)

    def test_zero_strategy(self, sample_df):
        """Test zero fill strategy."""
        imputer = FastImputer(strategy='zero')
        result = imputer.fit_transform(sample_df)
        
        assert result['Price'].iloc[2] == 0.0
        assert result['Volume'].iloc[1] == 0.0


class TestFastImputerEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        imputer = FastImputer(strategy='invalid')
        df = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            imputer.fit(df)

    def test_missing_columns_raises_error(self):
        """Test that specifying non-existent columns raises error."""
        imputer = FastImputer(strategy='mean', columns=['NonExistent'])
        df = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
        
        with pytest.raises(ValueError, match="Columns not found"):
            imputer.fit(df)

    def test_all_nan_column(self):
        """Test handling of column with all NaN values."""
        imputer = FastImputer(strategy='mean')
        df = pd.DataFrame({'A': [np.nan, np.nan, np.nan]})
        
        result = imputer.fit_transform(df)
        # Mean of all NaN is NaN, so result should still be NaN
        assert result['A'].isna().all()

    def test_no_nan_values(self):
        """Test that DataFrame without NaN values passes through unchanged."""
        imputer = FastImputer(strategy='linear')
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        
        result = imputer.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        imputer = FastImputer(strategy='linear')
        df = pd.DataFrame({'A': [np.nan]})
        
        result = imputer.fit_transform(df)
        # Linear interpolation can't fill single NaN, remains NaN
        assert result['A'].isna().all()

    def test_column_selection(self):
        """Test that only specified columns are imputed."""
        imputer = FastImputer(strategy='zero', columns=['A'])
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, np.nan, 6.0]
        })
        
        result = imputer.fit_transform(df)
        assert result['A'].iloc[1] == 0.0  # Imputed
        assert np.isnan(result['B'].iloc[1])  # Not imputed


class TestFastImputerSklearnCompat:
    """Test Scikit-learn compatibility."""

    def test_fit_returns_self(self):
        """Test that fit() returns self for method chaining."""
        imputer = FastImputer(strategy='mean')
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        
        result = imputer.fit(df)
        assert result is imputer

    def test_check_is_fitted(self):
        """Test that transform fails before fit."""
        imputer = FastImputer(strategy='mean')
        df = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
        
        # After fit, transform should work
        imputer.fit(df)
        result = imputer.transform(df)  # Should not raise
        assert not result['A'].isna().any()

    def test_fit_transform_equivalence(self):
        """Test fit_transform equals fit then transform."""
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, np.nan, 6.0]
        })
        
        imputer1 = FastImputer(strategy='mean')
        result1 = imputer1.fit_transform(df)
        
        imputer2 = FastImputer(strategy='mean')
        imputer2.fit(df)
        result2 = imputer2.transform(df)
        
        pd.testing.assert_frame_equal(result1, result2)

    def test_transform_preserves_dataframe_type(self):
        """Test that transform returns DataFrame, not ndarray."""
        imputer = FastImputer(strategy='mean')
        df = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
        
        result = imputer.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
