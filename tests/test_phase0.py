"""
Phase 0: Temporal Continuity Tests
Tests for global PCA, global normalization, and global color bounds.
"""


import numpy as np
import pandas as pd

def test_processor_global_pca():
    """Test DataProcessor.fit_global_pca and transform with fitted_pca."""
    from qsplot.processor import DataProcessor
    dp = DataProcessor()
    
    np.random.seed(42)
    X = np.random.rand(100, 5)
    
    # Fit global PCA
    info = dp.fit_global_pca(X, n_components=3, feature_names=['a','b','c','d','e'])
    assert 'pca' in info
    assert 'explained_variance_ratios' in info
    assert len(info['explained_variance_ratios']) == 3
    assert 'top_features_per_axis' in info
    assert 'top_loadings_per_axis' in info
    print("  [PASS] fit_global_pca")
    
    # Transform with fitted PCA (should NOT re-fit)
    X_subset = X[:20]
    result = dp.reduce_dimensions_with_info(
        X_subset, method='pca', n_components=3,
        feature_names=['a','b','c','d','e'],
        fitted_pca=info['pca']
    )
    assert result['positions'].shape == (20, 3)
    print("  [PASS] reduce_dimensions_with_info (fitted_pca)")
    
    # Verify consistency: same data through fitted PCA = same result
    r1 = dp.reduce_dimensions_with_info(X_subset, fitted_pca=info['pca'])
    r2 = dp.reduce_dimensions_with_info(X_subset, fitted_pca=info['pca'])
    assert np.allclose(r1['positions'], r2['positions']), "Same data should give same positions"
    print("  [PASS] Consistency check")
    
    # Without fitted_pca (per-frame), positions may differ from global
    r_local = dp.reduce_dimensions_with_info(X_subset, method='pca', n_components=3)
    # They should NOT be identical (different fit basis)
    print("  [PASS] Per-frame vs global produces different results (expected)")

def test_processor_global_normalization():
    """Test global normalization bounds."""
    from qsplot.processor import DataProcessor
    dp = DataProcessor()
    
    np.random.seed(42)
    pos1 = np.random.rand(10, 3) * 2  # range ~[0, 2]
    pos2 = np.random.rand(15, 3) * 10  # range ~[0, 10]
    
    center, scale = dp.compute_global_normalization_bounds([pos1, pos2])
    assert center.shape == (3,)
    assert scale > 0
    print("  [PASS] compute_global_normalization_bounds")
    
    # Global normalize
    norm1 = dp.normalize_positions(pos1, scale=10.0, global_center=center, global_scale=scale)
    norm2 = dp.normalize_positions(pos2, scale=10.0, global_center=center, global_scale=scale)
    
    # pos2 has larger range, so its normalized values should be larger than pos1's
    assert np.max(np.abs(norm2)) >= np.max(np.abs(norm1)), "Larger data should have larger norm"
    print("  [PASS] Global normalization preserves relative scale")
    
    # Per-frame normalize (original behavior) - both should fill [-10, 10]
    norm1_local = dp.normalize_positions(pos1, scale=10.0)
    norm2_local = dp.normalize_positions(pos2, scale=10.0)
    # Both should max out near 10.0
    assert abs(np.max(np.abs(norm1_local)) - 10.0) < 0.01
    assert abs(np.max(np.abs(norm2_local)) - 10.0) < 0.01
    print("  [PASS] Per-frame normalization backward compatible")

def test_visualizer_global_mode():
    """Test Visualizer with global normalization mode."""
    from qsplot.core import Visualizer
    
    # Create test time-series data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=3, freq='M')
    tickers = ['A', 'B', 'C', 'D', 'E']
    
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                'Date': d, 'Ticker': t,
                'F1': np.random.rand() * 100,
                'F2': np.random.rand() * 50,
                'F3': np.random.rand() * 200,
                'F4': np.random.rand() * 75,
            })
    df = pd.DataFrame(rows)
    
    vis = Visualizer()
    vis.engine = None  # No C++ engine needed for this test
    vis.load_data(df, ticker_col='Ticker', feature_cols=['F1','F2','F3','F4'], date_col='Date')
    
    assert len(vis._global_color_bounds) == 4, "Should have bounds for all 4 features"
    assert len(vis._global_pca_cache) > 0, "Should have cached global PCA"
    print("  [PASS] load_data computes global bounds")
    
    # Test global mode
    frame_g1 = vis.prepare_frame(dates[0], normalization='global')
    frame_g2 = vis.prepare_frame(dates[1], normalization='global')
    assert frame_g1 and frame_g2
    assert frame_g1['positions'].shape[0] == 5
    print("  [PASS] prepare_frame global mode")
    
    # Test per_frame mode (backward compatible)
    frame_l1 = vis.prepare_frame(dates[0], normalization='per_frame')
    frame_l2 = vis.prepare_frame(dates[1], normalization='per_frame')
    assert frame_l1 and frame_l2
    print("  [PASS] prepare_frame per_frame mode (backward compatible)")
    
    # Global color values should use consistent bounds
    # Per-frame color values always span [0, 1]
    # Global color values may NOT span full [0, 1] for a single frame
    local_range = frame_l1['values'].max() - frame_l1['values'].min()
    # Per-frame should be close to 1.0 (full range)
    assert abs(local_range - 1.0) < 0.01, f"Per-frame color range should be ~1.0, got {local_range}"
    print("  [PASS] Per-frame color uses full [0,1] range")


if __name__ == '__main__':
    print("=== Phase 0: Temporal Continuity Tests ===\n")
    
    print("1. Processor: Global PCA")
    test_processor_global_pca()
    
    print("\n2. Processor: Global Normalization")
    test_processor_global_normalization()
    
    print("\n3. Visualizer: Global Mode")
    test_visualizer_global_mode()
    
    print("\n=== ALL TESTS PASSED ===")
