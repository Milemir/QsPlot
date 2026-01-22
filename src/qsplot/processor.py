import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union

# Optional import for UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator

class DataProcessor:
    """
    Handles data cleaning, dimensionality reduction, and normalization
    for QsPlot.
    """
    
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handling missing values in the DataFrame.
        
        Args:
            df: Input DataFrame.
            strategy: 'mean', 'zero', 'drop', 'ffill'.
            
        Returns:
            Cleaned DataFrame.
        """
        if strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'zero':
            return df.fillna(0)
        elif strategy == 'drop':
            return df.dropna()
        elif strategy == 'ffill':
            return df.ffill().fillna(0) # ffill then 0 for leading NaNs
        else:
            raise ValueError(f"Unknown cleaning strategy: {strategy}")

    def reduce_dimensions(self, data: np.ndarray, method: str = 'pca', n_components: int = 3) -> np.ndarray:
        """
        Reduces dimensionality of the input data to n_components (usually 3 for xyz).
        
        Args:
            data: Input numpy array (N_samples, N_features).
            method: 'pca', 'tsne', 'umap'.
            n_components: Target dimensionality.
            
        Returns:
            Reduced numpy array (N_samples, n_components).
        """
        result = self.reduce_dimensions_with_info(data, method, n_components)
        return result['positions']
    
    def reduce_dimensions_with_info(self, data: np.ndarray, method: str = 'pca', 
                                     n_components: int = 3, 
                                     feature_names: Optional[List[str]] = None) -> dict:
        """
        Reduces dimensionality and returns additional info for UI display.
        
        Args:
            data: Input numpy array (N_samples, N_features).
            method: 'pca', 'tsne', 'umap'.
            n_components: Target dimensionality.
            feature_names: Optional list of feature column names.
            
        Returns:
            Dict with 'positions', 'explained_variance_ratios', 'top_features_per_axis'.
        """
        # Default feature names if not provided
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(data.shape[1])]
        
        # If data is already correct dimension (or less), just pad/return
        if data.shape[1] <= n_components:
            if data.shape[1] == n_components:
                positions = data
            else:
                # Pad with zeros
                positions = np.zeros((data.shape[0], n_components))
                positions[:, :data.shape[1]] = data
            return {
                'positions': positions,
                'explained_variance_ratios': [1.0 / n_components] * n_components,
                'top_features_per_axis': [[feature_names[i % len(feature_names)]] for i in range(n_components)]
            }

        reducer: BaseEstimator
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            positions = reducer.fit_transform(data)
            
            # Get explained variance ratios
            explained_var = reducer.explained_variance_ratio_.tolist()
            
            # Get top 3 features per component with their loadings
            top_features = []
            top_loadings = []
            components = reducer.components_  # (n_components, n_features)
            for i in range(n_components):
                loadings = np.abs(components[i])
                top_indices = np.argsort(loadings)[::-1][:3]  # Top 3
                top_names = [feature_names[idx] for idx in top_indices]
                top_vals = [loadings[idx] for idx in top_indices]
                top_features.append(top_names)
                top_loadings.append(top_vals)
            
            return {
                'positions': positions,
                'explained_variance_ratios': explained_var,
                'top_features_per_axis': top_features,
                'top_loadings_per_axis': top_loadings
            }
            
        elif method == 'tsne':
            # t-SNE has no explained variance concept
            reducer = TSNE(n_components=n_components)
            positions = reducer.fit_transform(data)
            return {
                'positions': positions,
                'explained_variance_ratios': None,
                'top_features_per_axis': None
            }
            
        elif method == 'umap':
            if not HAS_UMAP:
                print("Warning: UMAP not installed, falling back to PCA.")
                return self.reduce_dimensions_with_info(data, 'pca', n_components, feature_names)
            else:
                reducer = umap.UMAP(n_components=n_components)
                positions = reducer.fit_transform(data)
                return {
                    'positions': positions,
                    'explained_variance_ratios': None,
                    'top_features_per_axis': None
                }
        else:
            raise ValueError(f"Unknown reduction method: {method}")

    def normalize_positions(self, data: np.ndarray, scale: float = 10.0) -> np.ndarray:
        """
        Centers and scales data to fit within [-scale, scale].
        
        Args:
            data: Input positions (N, 3).
            scale: Max bound for the cube.
            
        Returns:
            Normalized data.
        """
        # Center
        centered = data - np.mean(data, axis=0)
        
        # Scale max abs value to 1.0 then multiply by scale
        max_val = np.max(np.abs(centered))
        if max_val > 0:
            centered = centered / max_val
            
        return centered * scale
