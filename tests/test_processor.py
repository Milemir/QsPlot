"""
Tests for DataProcessor - data cleaning, dimensionality reduction, and normalization.
"""

import numpy as np
import pandas as pd
import pytest

from qsplot.processor import DataProcessor


class TestDataProcessorCleanData:
    """Test data cleaning strategies."""

    @pytest.fixture
    def processor(self):
        return DataProcessor()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10.0, 20.0, np.nan, 40.0, 50.0],
            'C': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

    def test_clean_data_mean(self, processor, sample_df):
        """Test mean filling strategy."""
        result = processor.clean_data(sample_df, strategy='mean')
        
        # NaN should be replaced with column mean
        assert not result.isna().any().any()
        # Check A column mean was applied
        expected_mean_a = sample_df['A'].mean()
        assert result['A'].iloc[1] == pytest.approx(expected_mean_a)

    def test_clean_data_zero(self, processor, sample_df):
        """Test zero filling strategy."""
        result = processor.clean_data(sample_df, strategy='zero')
        
        assert result['A'].iloc[1] == 0.0
        assert result['B'].iloc[2] == 0.0

    def test_clean_data_drop(self, processor, sample_df):
        """Test drop rows strategy."""
        result = processor.clean_data(sample_df, strategy='drop')
        
        # Only rows 0, 4 have no NaN (rows 2, 4 in original - C column is complete)
        # Actually: row 0 (A=1, B=10), row 4 (A=5, B=50) - no NaN
        assert len(result) == 2
        assert not result.isna().any().any()

    def test_clean_data_ffill(self, processor, sample_df):
        """Test forward fill strategy."""
        result = processor.clean_data(sample_df, strategy='ffill')
        
        # A[1] should be 1.0 (previous value)
        assert result['A'].iloc[1] == 1.0
        # B[2] should be 20.0 (previous value)
        assert result['B'].iloc[2] == 20.0

    def test_clean_data_invalid_strategy(self, processor, sample_df):
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown cleaning strategy"):
            processor.clean_data(sample_df, strategy='invalid')


class TestDataProcessorReduceDimensions:
    """Test dimensionality reduction methods."""

    @pytest.fixture
    def processor(self):
        return DataProcessor()

    @pytest.fixture
    def high_dim_data(self):
        """Create high-dimensional data (100 samples, 10 features)."""
        np.random.seed(42)
        return np.random.randn(100, 10)

    def test_pca_reduction(self, processor, high_dim_data):
        """Test PCA reduces to correct dimension."""
        result = processor.reduce_dimensions(high_dim_data, method='pca', n_components=3)
        
        assert result.shape == (100, 3)

    def test_pca_single_component(self, processor, high_dim_data):
        """Test PCA to 1 dimension."""
        result = processor.reduce_dimensions(high_dim_data, method='pca', n_components=1)
        
        assert result.shape == (100, 1)

    def test_tsne_reduction(self, processor):
        """Test t-SNE reduction (use smaller dataset for speed)."""
        np.random.seed(42)
        small_data = np.random.randn(31, 5)
        
        result = processor.reduce_dimensions(small_data, method='tsne', n_components=2)
        
        assert result.shape == (31, 2)

    def test_data_already_correct_dims(self, processor):
        """Test data that already has target dimensions."""
        data = np.random.randn(50, 3)
        
        result = processor.reduce_dimensions(data, method='pca', n_components=3)
        
        # Should return as-is
        np.testing.assert_array_equal(result, data)

    def test_data_fewer_dims_than_target(self, processor):
        """Test data with fewer dimensions than target - should pad with zeros."""
        data = np.random.randn(50, 2)
        
        result = processor.reduce_dimensions(data, method='pca', n_components=3)
        
        assert result.shape == (50, 3)
        # Third column should be zeros
        np.testing.assert_array_equal(result[:, 2], np.zeros(50))

    def test_invalid_method(self, processor, high_dim_data):
        """Test invalid reduction method raises error."""
        with pytest.raises(ValueError, match="Unknown reduction method"):
            processor.reduce_dimensions(high_dim_data, method='invalid')


class TestDataProcessorNormalizePositions:
    """Test position normalization."""

    @pytest.fixture
    def processor(self):
        return DataProcessor()

    def test_normalize_default_scale(self, processor):
        """Test normalization to default scale [-10, 10]."""
        data = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 200.0, 300.0],
            [-50.0, 50.0, 0.0]
        ])
        
        result = processor.normalize_positions(data, scale=10.0)
        
        # Max absolute value should be 10.0
        assert np.max(np.abs(result)) == pytest.approx(10.0)

    def test_normalize_custom_scale(self, processor):
        """Test normalization with custom scale."""
        data = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        
        result = processor.normalize_positions(data, scale=5.0)
        
        assert np.max(np.abs(result)) == pytest.approx(5.0)

    def test_normalize_centers_data(self, processor):
        """Test that normalization centers data around origin."""
        data = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 40.0, 60.0],
            [30.0, 60.0, 90.0]
        ])
        
        result = processor.normalize_positions(data)
        
        # Mean should be at origin
        np.testing.assert_array_almost_equal(
            np.mean(result, axis=0), 
            np.zeros(3), 
            decimal=10
        )

    def test_normalize_zero_data(self, processor):
        """Test normalization of all-zero data."""
        data = np.zeros((10, 3))
        
        result = processor.normalize_positions(data)
        
        # Should remain zeros (avoid division by zero)
        np.testing.assert_array_equal(result, np.zeros((10, 3)))

    def test_normalize_single_point(self, processor):
        """Test normalization of single point."""
        data = np.array([[5.0, 10.0, 15.0]])
        
        result = processor.normalize_positions(data)
        
        # Single point centered = all zeros
        np.testing.assert_array_equal(result, np.zeros((1, 3)))
