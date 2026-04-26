# QsPlot
**High-Performance Visualization for Time Series Datasets**

QsPlot combines a C++20/OpenGL backend with a Python/Pandas frontend to analyze and visualize large-dimensional time-series data.


https://github.com/user-attachments/assets/d0d1113b-2c42-4f0a-b80d-c585f8726800


## Documentation

- **[User Guide](infra/docs/user_guide.md)**: Installation, tutorials, and concepts.
- **[API Reference](infra/docs/api_reference.md)**: Details on `Visualizer` and `DataProcessor`.

## Quick Start

### 1. Installation

**Prerequisites**: CMake 3.15+, C++ Compiler (MSVC/GCC/Clang), Python 3.8+.

```bash
# Clone the repo (and its submodules)
git clone --recursive https://github.com/Milemir/QsPlot.git
cd QsPlot

# Install via Pip (Builds from source)
pip install .
```

### 2.1 Using `animate()` for Time Series Data

```python
from qsplot import Visualizer
import pandas as pd

# Load your data
df = pd.read_csv("data.csv") 

# Initialize
vis = Visualizer()

# Ingest data (use date_col for time series)
vis.load_data(df, date_col="Date", ticker_col="Ticker", feature_cols=['Feature_1', 'Feature_2'])

# Animate across the time series
vis.animate("2024-01-01", "2024-12-31")
```

### 2.2 Advanced: ML Integration & Selection Export

QsPlot can natively run Machine Learning models (like K-Means and Isolation Forest) on your dataset, add the generated labels as visual features, and allow you to select and export data ranges visually.

```python
# 1. Load Data
vis = Visualizer()
vis.load_data(df, ticker_col='Ticker', feature_cols=['F1', 'F2', 'F3'], date_col='Date')

# 2. Machine Learning
# Compute clusters globally (adds 'Cluster' to features)
vis.compute_clusters(n_clusters=5, method='kmeans')

# Detect anomalies globally (adds 'Outlier_Score' to features)
vis.compute_outliers(contamination=0.05, method='isolation_forest')

# 3. Visualize & Interact
vis.animate("2024-01-01", "2024-12-31")
# -> In the UI: Select 'Cluster' or 'Outlier_Score' from the Color Feature dropdown.
# -> In the UI: Hold SHIFT + Left Click & Drag to draw a rectangle selection.

# 4. Export the data you selected visually in the UI
df_selected = vis.get_selected_points("2024-06-15")
vis.export_selection("my_anomalies.csv", date="2024-06-15")
```


### 2.3 Using `static()` for Single Snapshots

The `static()` method has two main use cases:

**A) View a specific timestamp from time series data:**

```python
from qsplot import Visualizer
import pandas as pd

# Load time series data
df = pd.read_csv("data.csv")

vis = Visualizer()
vis.load_data(df, date_col='Date', ticker_col='Ticker', 
              feature_cols=['Feature_1', 'Feature_2', 'Feature_3'])

# View a specific date without animation
vis.static("2024-06-15")
```

**B) Visualize Static Data:**
```python
from qsplot import Visualizer
import pandas as pd

# Load static data
df = pd.read_csv("data.csv")

vis = Visualizer()
vis.load_data(df, ticker_col='Ticker',
              feature_cols=['Feature_1', 'Feature_2', 'Feature_3'])

# Display static visualization (auto-uses the single date)
vis.static()
```

## Setup Dependencies (Manual)

If `pip install` fails due to missing submodules, you can run the helper script:

```powershell
infra/scripts/setup_dependencies.ps1
```

This will download dependencies into `infra/dep`.
