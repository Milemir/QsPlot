
import numpy as np
import time
from .qsplot_engine import Renderer, DataProcessor

class QsPlotWrapper:
    """
    High-level Python wrapper for the QsPlot C++ library.
    
    This class manages the Renderer and DataProcessor, providing a simplified
    API for loading datasets with variable features and different dimensionality 
    reduction techniques.
    """

    def __init__(self):
        """Initialize the QsPlot engine components."""
        self.renderer = Renderer()
        self.processor = DataProcessor()
        self.is_running = False

    def start(self):
        """Start the background rendering thread."""
        if not self.is_running:
            self.renderer.start()
            self.is_running = True

    def stop(self):
        """Stop the background rendering thread."""
        if self.is_running:
            self.renderer.stop()
            self.is_running = False

    def load_dataset(self, data, method='pca', feature_col=None, target_dims=3):
        """
        Load a dataset into the visualization engine.

        Args:
            data (np.ndarray): Input data of shape (N_samples, N_features).
            method (str): Dimensionality reduction method.
                          - 'pca': Use internal C++ PCA (Default).
                          - 'raw': Use data as-is (Must be N x 3).
                          - 'none': Alias for 'raw'.
            feature_col (int, optional): Column index to use for point values (color/size).
                                         If None, uses 1.0 (uniform).
            target_dims (int): Number of dimensions to reduce to (Default: 3 for 3D view).
        """
        # Ensure input is a numpy array
        data = np.asarray(data)
        N, M = data.shape

        points_3d = None
        
        # 1. Coordinate Processing
        if method.lower() == 'pca':
            # DataProcessor expects MatrixXd (double)
            # We explicitly cast to float64 to match Eigen::MatrixXd
            data_double = data.astype(np.float64)
            
            print(f"[QsPlot] Computing PCA on {N} samples, {M} features -> {target_dims} dims...")
            self.processor.load_data(data_double)
            
            # compute_pca returns a copy, likely double
            points_eigen = self.processor.compute_pca(target_dims)
            
            # Renderer expects float32
            points_3d = np.array(points_eigen, dtype=np.float32)
            
        elif method.lower() in ['raw', 'none']:
            if M != 3:
                raise ValueError(f"For method='{method}', data must have exactly 3 columns (got {M}).")
            points_3d = data.astype(np.float32)
            
        else:
            raise ValueError(f"Unknown reduction method: '{method}'. Supported: 'pca', 'raw'.")

        # 2. Value/Feature Processing
        values = np.ones(N, dtype=np.float32)
        
        if feature_col is not None:
            if isinstance(feature_col, int):
                if 0 <= feature_col < M:
                    col_data = data[:, feature_col]
                    values = col_data.astype(np.float32)
                else:
                    print(f"[QsPlot] Warning: feature_col {feature_col} out of bounds.")
            elif hasattr(feature_col, "__len__") and len(feature_col) == N:
                values = np.array(feature_col, dtype=np.float32)

        # 3. Upload to Render Engine
        self.renderer.set_points(points_3d, values)
        print(f"[QsPlot] Uploaded {N} points to renderer.")

    def get_explained_variance(self):
        """Get the explained variance ratio from the last PCA run."""
        return self.processor.get_explained_variance_ratio()
