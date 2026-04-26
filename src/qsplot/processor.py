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
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

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

    def fit_global_pca(self, data: np.ndarray, n_components: int = 3, 
                        feature_names: Optional[List[str]] = None) -> dict:
        """
        Fit PCA on the full dataset (all timestamps concatenated) and return
        the fitted model + metadata. This fitted PCA can then be passed to
        reduce_dimensions_with_info() for consistent axes across frames.
        
        Args:
            data: Full dataset (all timestamps), shape (N_total, N_features).
            n_components: Target dimensionality.
            feature_names: Optional feature column names.
            
        Returns:
            Dict with 'pca' (fitted PCA object), 'explained_variance_ratios',
            'top_features_per_axis', 'top_loadings_per_axis'.
        """
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(data.shape[1])]
        
        pca = PCA(n_components=n_components)
        pca.fit(data)
        
        info = self._extract_pca_info(pca, feature_names, n_components)
        info['pca'] = pca
        return info
    
    def _extract_pca_info(self, pca: PCA, feature_names: List[str], 
                           n_components: int) -> dict:
        """
        Extract explained variance and top feature loadings from a fitted PCA.
        """
        explained_var = pca.explained_variance_ratio_.tolist()
        
        top_features = []
        top_loadings = []
        components = pca.components_  # (n_components, n_features)
        for i in range(n_components):
            loadings = np.abs(components[i])
            top_indices = np.argsort(loadings)[::-1][:3]  # Top 3
            top_names = [feature_names[idx] for idx in top_indices]
            top_vals = [loadings[idx] for idx in top_indices]
            top_features.append(top_names)
            top_loadings.append(top_vals)
        
        return {
            'explained_variance_ratios': explained_var,
            'top_features_per_axis': top_features,
            'top_loadings_per_axis': top_loadings
        }

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
                                     feature_names: Optional[List[str]] = None,
                                     fitted_pca: Optional[PCA] = None) -> dict:
        """
        Reduces dimensionality and returns additional info for UI display.
        
        Args:
            data: Input numpy array (N_samples, N_features).
            method: 'pca', 'tsne', 'umap'.
            n_components: Target dimensionality.
            feature_names: Optional list of feature column names.
            fitted_pca: Optional pre-fitted PCA object. If provided, only
                         transform() is called (no fit), ensuring axis consistency
                         across multiple calls (e.g., animation frames).
            
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
            if fitted_pca is not None:
                # Use pre-fitted PCA: only transform, no fit
                # This ensures axes are consistent across frames
                positions = fitted_pca.transform(data)
                info = self._extract_pca_info(fitted_pca, feature_names, n_components)
            else:
                # Per-frame fit (original behavior)
                reducer = PCA(n_components=n_components)
                positions = reducer.fit_transform(data)
                info = self._extract_pca_info(reducer, feature_names, n_components)
            
            return {
                'positions': positions,
                **info
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

    def normalize_positions(self, data: np.ndarray, scale: float = 10.0,
                             global_center: Optional[np.ndarray] = None,
                             global_scale: Optional[float] = None) -> np.ndarray:
        """
        Centers and scales data to fit within [-scale, scale].
        
        Args:
            data: Input positions (N, 3).
            scale: Max bound for the cube.
            global_center: Optional pre-computed center (mean) for consistent
                           centering across frames. If None, uses per-frame mean.
            global_scale: Optional pre-computed max absolute value for consistent
                          scaling across frames. If None, uses per-frame max.
            
        Returns:
            Normalized data.
        """
        # Center
        center = global_center if global_center is not None else np.mean(data, axis=0)
        centered = data - center
        
        # Scale max abs value to 1.0 then multiply by scale
        max_val = global_scale if global_scale is not None else np.max(np.abs(centered))
        if max_val > 0:
            centered = centered / max_val
            
        return centered * scale

    def compute_global_normalization_bounds(self, all_positions: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Compute global center and scale from a list of position arrays
        (one per timestamp). Use the results with normalize_positions()
        for consistent normalization across animation frames.
        
        Args:
            all_positions: List of (N_i, 3) arrays, one per timestamp.
            
        Returns:
            Tuple of (global_center (3,), global_scale (float)).
        """
        all_pos = np.vstack(all_positions)
        global_center = np.mean(all_pos, axis=0)
        global_scale = np.max(np.abs(all_pos - global_center))
        return global_center, global_scale

    # --- Phase 3: ML Integration ---
    
    def compute_clusters(self, X: np.ndarray, n_clusters: int = 5, method: str = 'kmeans') -> np.ndarray:
        """
        Computes cluster assignments for the given feature matrix.
        
        Args:
            X: Feature matrix (N, F).
            n_clusters: Number of clusters to find.
            method: Clustering algorithm (currently only 'kmeans' is supported).
            
        Returns:
            Cluster labels (N,).
        """
        if method.lower() == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            return model.fit_predict(X)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

    def detect_outliers(self, X: np.ndarray, contamination: float = 0.05, method: str = 'isolation_forest') -> np.ndarray:
        """
        Detects outliers in the feature matrix and returns anomaly scores.
        Lower scores mean more abnormal (standard Isolation Forest behavior).
        This function normalizes the scores to [0, 1] where 1 is the most anomalous
        for better visualization mapping.
        
        Args:
            X: Feature matrix (N, F).
            contamination: Expected proportion of outliers.
            method: Outlier detection algorithm (currently only 'isolation_forest' is supported).
            
        Returns:
            Anomaly scores in range [0, 1] where 1 is most anomalous.
        """
        if method.lower() == 'isolation_forest':
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(X)
            # decision_function returns lower scores for outliers, higher for inliers.
            scores = model.decision_function(X)
            # Normalize to [0, 1] where 1 is the most anomalous (lowest original score)
            scores_norm = 1.0 - ((scores - scores.min()) / (scores.max() - scores.min()))
            return scores_norm
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
