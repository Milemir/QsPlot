# API Reference

## `qsplot.Visualizer`

The high-level controller for the visualization engine.

### `__init__(self)`
Initializes the C++ Renderer and starts the GUI thread.

### `load_data(self, df, ticker_col, feature_cols, date_col=None, freq='M', missing_strategy='mean')`
Ingests a DataFrame for analysis. Works for both time series and non-temporal data.
- **df** (`pd.DataFrame`): Source data.
- **ticker_col** (`str`): Name of the identifier column (ticker, item ID, etc.).
- **feature_cols** (`List[str]`): List of columns to use for dimensionality reduction.
- **date_col** (`str`, optional): Name of the date column. If None, auto-creates a date column (for static data).
- **freq** (`str`): Frequency tag (metadata only).
- **missing_strategy** (`str`): How to clean missing data ('mean', 'zero', 'drop', 'ffill').

### `load_time_series(...)` 
**Deprecated:** Use `load_data()` instead. This method is kept for backward compatibility.

### `prepare_frame(self, date, method='pca', n_components=3) -> Dict`
internal or advanced usage. Process a single timestamp.
- **Returns**: Dictionary with keys `positions` (Nx3), `values` (Nx1), `tickers` (N).

### `animate(self, start_date, end_date, method='pca')`
Runs a blocking animation loop from start to end date for time series data.
- **method** (`str`): 'pca', 'tsne', or 'umap'.

### `static(self, date=None, method='pca')`
Displays a static (non-animated) visualization for a single timestamp. Perfect for non-time series data or viewing a single snapshot.
- **date** (`str`, optional): Specific date to visualize. If None, uses the first available date.
- **method** (`str`): 'pca', 'tsne', or 'umap'.

**Example:**
```python
# For non-time series data
vis.static()

# For specific date
vis.static("2024-06-15")
```

### `stop(self)`
Stops the engine and closes the window.

---

## `qsplot.DataProcessor`

Math and cleaning utilities.

### `clean_data(self, df, strategy='mean') -> pd.DataFrame`
Fills or drops NaNs based on strategy.

### `reduce_dimensions(self, data, method='pca', n_components=3) -> np.ndarray`
Projects `(N, F)` data down to `(N, 3)`.
- **data**: Numpy array of shape (Samples, Features).
- **method**: 'pca', 'tsne', 'umap'.

### `normalize_positions(self, data, scale=10.0) -> np.ndarray`
Centers data at (0,0,0) and scales max absolute value to `scale`.
