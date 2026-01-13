# QsPlot

**High-Performance Visualization for Quantitative Finance**

QsPlot combines a C++20/OpenGL backend with a Python/Pandas frontend to analyze and visualize large-dimensional time-series data.

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

### 2. Run an Analysis

```python
from qsplot import Visualizer
import pandas as pd

# Load your data
df = pd.read_csv("finance_data.csv") 

# Initialize
vis = Visualizer()

# Ingest
vis.load_time_series(df, date_col="Date", ticker_col="Ticker", feature_cols=["F1", "F2", "F3"])

# Animate (time series)
vis.run_morph_animation("2024-01-01", "2024-12-31")

# OR: Static visualization (single snapshot)
vis.run_static_visualization()
```

## Setup Dependencies (Manual)

If `pip install` fails due to missing submodules, you can run the helper script:

```powershell
infra/scripts/setup_dependencies.ps1
```

This will download dependencies into `infra/dep`.
