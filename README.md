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

# Ingest
vis.load_time_series(df, date_col="Date", ticker_col="Ticker", feature_cols=["F1", "F2", "F3"])

# Animate (time series)
vis.animate("2024-01-01", "2024-12-31")

# OR: Static visualization (single snapshot)
vis.static()
```

### 2.2 Using `static()` for Non-Time Series Data

For datasets without temporal progression or when you want to view a single snapshot:

```python
from qsplot import Visualizer
import pandas as pd

# Load your data
df = pd.read_csv("data.csv")

# Initialize
vis = Visualizer()

# Ingest
vis.load_time_series(df, date_col='Date', ticker_col='Ticker', feature_cols=['Feature_1', 'Feature_2', ...])

# Display static 3D visualization
vis.static()  # Uses first available date

# Or specify a particular date
vis.static("2024-06-15")
```

## Setup Dependencies (Manual)

If `pip install` fails due to missing submodules, you can run the helper script:

```powershell
infra/scripts/setup_dependencies.ps1
```

This will download dependencies into `infra/dep`.
