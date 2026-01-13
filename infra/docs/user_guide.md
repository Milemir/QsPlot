# QsPlot User Guide

## Introduction

QsPlot describes a high-performance hybrid visualization system:
- **Backend**: C++20 / OpenGL 4.1 for rendering millions of points efficiently.
- **Frontend**: Python for data science/Pandas integration.

## Installation

### Prerequisites
1.  **C++ Compiler**: MSVC 2019+, GCC 10+, or Clang 11+.
2.  **CMake**: Version 3.15 or newer.
3.  **Python**: 3.8+.

### Installing via Pip

The easiest way to install is to build from source using pip. This compiles the C++ backend and installs the Python package.

```bash
pip install .
```

*Note: This process may take a few minutes as it compiles dependencies like Nanobind and the engine itself.*

## Core Concepts

### 1. The Visualizer (`qsplot.Visualizer`)
The `Visualizer` is your main entry point. It orchestrates the entire workflow:
- **Ingestion**: Takes a Pandas DataFrame.
- **Processing**: Cleans, reduces dimensions, and normalizes data.
- **Rendering**: Sends processed data to the C++ Engine.
- **Animation**: Manages the "Morphing" between time steps.

### 2. The Data Processor (`qsplot.DataProcessor`)
A utility class handling math operations:
- **PCA/UMAP**: Dimensionality reduction to map high-dimensional financial data to 3D space.
- **Normalization**: Scales point clouds to fit within the visible 3D world (default [-10, 10]).

### 3. The Ticker Alignment Problem
A key feature of `qsplot` is handling time-series where companies (tickers) appear and disappear.
When morphing from **Month A** to **Month B**:
- The engine finds the **intersection** of tickers present in both months.
- Only these common tickers are morphed physically.
- This ensures visual continuity.

## Tutorial: Detailed Analysis

See `examples/main_analysis.py` for a runnable example.

### Step 1: Prepare Data
Your DataFrame should look like this:

| Date       | Ticker | Feat_0 | Feat_1 | ... |
|------------|--------|--------|--------|-----|
| 2024-01-01 | AAPL   | 0.5    | 1.2    | ... |
| 2024-01-01 | MSFT   | 0.3    | 0.9    | ... |

### Step 2: Initialize
```python
from qsplot import Visualizer
vis = Visualizer()
```
*The OpenGL window will open immediately.*

### Step 3: Load
```python
vis.load_time_series(
    df=my_df,
    date_col="Date",
    ticker_col="Ticker",
    feature_cols=["Feat_0", "Feat_1", ...],
    freq='M'
)
```

### Step 4: Animate
```python
vis.animate(
    start_date="2024-01-01", 
    end_date="2024-12-31", 
    method='pca' # or 'umap'
)
```

## Static Visualization (Non-Time Series)

If you have data without meaningful temporal progression, or you just want to view a single snapshot, use `static()`:

```python
from qsplot import Visualizer
import pandas as pd

# Your data (can be any date, doesn't matter for static view)
df = pd.DataFrame({
    'Date': ['2024-01-01'] * 100,  # All same date
    'Ticker': [f'Stock_{i}' for i in range(100)],
    'Feature_1': np.random.randn(100),
    'Feature_2': np.random.randn(100),
    'Feature_3': np.random.randn(100),
})

vis = Visualizer()
vis.load_time_series(df, date_col='Date', ticker_col='Ticker', 
                      feature_cols=['Feature_1', 'Feature_2', 'Feature_3'])

# Display static visualization
vis.static()
```