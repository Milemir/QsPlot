"""
Tests for Visualizer - main orchestration class.
Uses mocking to avoid requiring the C++ engine during tests.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestVisualizerDataLoading:
    """Test data loading and preparation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample time-series DataFrame."""
        dates = pd.date_range("2024-01-01", periods=3, freq="ME")
        tickers = ["AAPL", "MSFT", "GOOG"]
        
        data = []
        for date in dates:
            for ticker in tickers:
                data.append({
                    "Date": date,
                    "Ticker": ticker,
                    "F1": np.random.randn(),
                    "F2": np.random.randn(),
                    "F3": np.random.randn(),
                    "F4": np.random.randn()
                })
        return pd.DataFrame(data)

    @patch('qsplot.core.qsplot_engine', None)
    def test_visualizer_init_without_engine(self):
        """Test Visualizer initializes without C++ engine."""
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        assert vis.engine is None
        assert vis.processor is not None

    @patch('qsplot.core.qsplot_engine')
    def test_load_time_series(self, mock_engine, sample_df):
        """Test load_time_series processes DataFrame correctly."""
        mock_engine.Renderer = Mock
        
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        vis.load_time_series(
            df=sample_df,
            date_col="Date",
            ticker_col="Ticker",
            feature_cols=["F1", "F2", "F3", "F4"]
        )
        
        assert vis.df is not None
        assert vis._date_col == "Date"
        assert vis._ticker_col == "Ticker"
        assert vis._feature_cols == ["F1", "F2", "F3", "F4"]

    @patch('qsplot.core.qsplot_engine')
    def test_get_dates(self, mock_engine, sample_df):
        """Test get_dates returns unique dates."""
        mock_engine.Renderer = Mock
        
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        vis.load_time_series(
            df=sample_df,
            date_col="Date",
            ticker_col="Ticker",
            feature_cols=["F1", "F2"]
        )
        
        dates = vis.get_dates()
        assert len(dates) == 3

    @patch('qsplot.core.qsplot_engine', None)
    def test_get_dates_no_data(self):
        """Test get_dates returns empty array when no data loaded."""
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        dates = vis.get_dates()
        
        assert len(dates) == 0


class TestVisualizerPrepareFrame:
    """Test frame preparation pipeline."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with multiple dates."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=2, freq="ME")
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        
        data = []
        for date in dates:
            for ticker in tickers:
                data.append({
                    "Date": date,
                    "Ticker": ticker,
                    "F1": np.random.randn() * 10 + 50,
                    "F2": np.random.randn() * 5 + 20,
                    "F3": np.random.randn() * 2 + 10,
                })
        return pd.DataFrame(data)

    @patch('qsplot.core.qsplot_engine', None)
    def test_prepare_frame_returns_dict(self, sample_df):
        """Test prepare_frame returns proper dictionary structure."""
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        vis.load_time_series(
            df=sample_df,
            date_col="Date",
            ticker_col="Ticker",
            feature_cols=["F1", "F2", "F3"]
        )
        
        result = vis.prepare_frame("2024-01-31")
        
        assert "positions" in result
        assert "values" in result
        assert "tickers" in result

    @patch('qsplot.core.qsplot_engine', None)
    def test_prepare_frame_correct_shape(self, sample_df):
        """Test prepare_frame returns correct array shapes."""
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        vis.load_time_series(
            df=sample_df,
            date_col="Date",
            ticker_col="Ticker",
            feature_cols=["F1", "F2", "F3"]
        )
        
        result = vis.prepare_frame("2024-01-31")
        
        # 4 tickers for this date
        assert result["positions"].shape == (4, 3)
        assert result["values"].shape == (4,)
        assert len(result["tickers"]) == 4

    @patch('qsplot.core.qsplot_engine', None)
    def test_prepare_frame_dtype_float32(self, sample_df):
        """Test prepare_frame returns float32 arrays for C++ compatibility."""
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        vis.load_time_series(
            df=sample_df,
            date_col="Date",
            ticker_col="Ticker",
            feature_cols=["F1", "F2", "F3"]
        )
        
        result = vis.prepare_frame("2024-01-31")
        
        assert result["positions"].dtype == np.float32
        assert result["values"].dtype == np.float32

    @patch('qsplot.core.qsplot_engine', None)
    def test_prepare_frame_invalid_date(self, sample_df):
        """Test prepare_frame with non-existent date returns empty dict."""
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        vis.load_time_series(
            df=sample_df,
            date_col="Date",
            ticker_col="Ticker",
            feature_cols=["F1", "F2", "F3"]
        )
        
        result = vis.prepare_frame("1999-01-01")  # Date not in data
        
        assert result == {}

    @patch('qsplot.core.qsplot_engine', None)
    def test_prepare_frame_values_normalized(self, sample_df):
        """Test that values are normalized to [0, 1] range."""
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        vis.load_time_series(
            df=sample_df,
            date_col="Date",
            ticker_col="Ticker",
            feature_cols=["F1", "F2", "F3"]
        )
        
        result = vis.prepare_frame("2024-01-31")
        
        # Values should be normalized between 0 and 1
        assert result["values"].min() >= 0.0
        assert result["values"].max() <= 1.0


class TestVisualizerTickerAlignment:
    """Test ticker alignment between frames."""

    @pytest.fixture
    def df_with_changing_tickers(self):
        """Create DataFrame where tickers change between dates."""
        data = [
            # January - AAPL, MSFT, GOOG
            {"Date": "2024-01-31", "Ticker": "AAPL", "F1": 1.0, "F2": 2.0},
            {"Date": "2024-01-31", "Ticker": "MSFT", "F1": 3.0, "F2": 4.0},
            {"Date": "2024-01-31", "Ticker": "GOOG", "F1": 5.0, "F2": 6.0},
            # February - MSFT, GOOG, AMZN (AAPL dropped, AMZN added)
            {"Date": "2024-02-29", "Ticker": "MSFT", "F1": 7.0, "F2": 8.0},
            {"Date": "2024-02-29", "Ticker": "GOOG", "F1": 9.0, "F2": 10.0},
            {"Date": "2024-02-29", "Ticker": "AMZN", "F1": 11.0, "F2": 12.0},
        ]
        return pd.DataFrame(data)

    @patch('qsplot.core.qsplot_engine', None)
    def test_ticker_alignment_finds_intersection(self, df_with_changing_tickers):
        """Test that only common tickers are morphed."""
        from qsplot.core import Visualizer
        
        vis = Visualizer()
        vis.load_time_series(
            df=df_with_changing_tickers,
            date_col="Date",
            ticker_col="Ticker",
            feature_cols=["F1", "F2"]
        )
        
        frame1 = vis.prepare_frame("2024-01-31")
        frame2 = vis.prepare_frame("2024-02-29")
        
        # Find common tickers
        common = np.intersect1d(frame1["tickers"], frame2["tickers"])
        
        # Should have MSFT and GOOG in common
        assert len(common) == 2
        assert "MSFT" in common
        assert "GOOG" in common
        assert "AAPL" not in common
        assert "AMZN" not in common
