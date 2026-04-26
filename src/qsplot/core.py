import numpy as np
import pandas as pd
import time
from typing import List, Optional, Union, Dict, Any
from .processor import DataProcessor

# Import the C++ Module
try:
    from . import qsplot_engine
except ImportError:
    try:
        import qsplot_engine
    except ImportError:
        print("Warning: qsplot_engine C++ module not found. Is it installed?")
        qsplot_engine = None

class Visualizer:
    """
    Orchestrates the visualization workflow:
    - Data ingestion
    - Dimensionality Reduction
    - Sending data to C++ Engine
    - Animation control
    """
    
    def __init__(self):
        if qsplot_engine:
            self.engine = qsplot_engine.Renderer()
            self.engine.start()
        else:
            self.engine = None
            
        self.processor = DataProcessor()
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        
        # Internal Cache
        self._date_col = None
        self._ticker_col = None
        self._feature_cols = None 
        
        # Global PCA Cache (for temporal continuity)
        self._global_pca_cache: Dict[str, Any] = {}  # keyed by color_feature
        self._global_color_bounds: Dict[str, tuple] = {}  # {col_name: (min, max)}
        self._global_norm_bounds: Dict[str, tuple] = {}  # {cache_key: (center, scale)}
        
    def load_data(self, 
                  df: pd.DataFrame, 
                  ticker_col: str,
                  feature_cols: List[str],
                  date_col: Optional[str] = None,
                  freq: str = 'M',
                  missing_strategy: str = 'mean'):
        """
        Loads and prepares the DataFrame for analysis.

        Args:
            df: Source DataFrame
            ticker_col: Name of the identifier column (ticker, item ID, etc.)
            feature_cols: List of feature column names for dimensionality reduction
            date_col: Name of the date column. If None, creates a date column automatically 
                     (useful for static/non-temporal data).
            freq: Frequency tag (metadata only, e.g., 'M', 'D')
            missing_strategy: How to handle missing values ('mean', 'zero', 'drop', 'ffill')
        """
        print(f"Loading {len(df)} rows...")
        self.df = df.copy()
        
        # Handle date column
        if date_col is None:
            # Auto-create date column for static data
            print("No date column specified. Creating default date column for static visualization...")
            date_col = '_qsplot_date_'
            self.df[date_col] = pd.Timestamp('2024-01-01')
        else:
            # Convert existing date column only if not already datetime
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # Sort by date
        self.df = self.df.sort_values(by=date_col)
        
        # Store metadata
        self._date_col = date_col
        self._ticker_col = ticker_col
        self._feature_cols = feature_cols
        self.metadata['freq'] = freq
        
        # Pre-cleaning
        print("Cleaning data...")
        self.df[feature_cols] = self.processor.clean_data(self.df[feature_cols], strategy=missing_strategy)
        print("Data loaded and cleaned.")
        
        # Pre-compute global color bounds for all features
        self._compute_global_color_bounds()
        
        # Pre-compute global PCA (default: auto color feature)
        # Additional PCA fits will be cached lazily when different color features are selected
        self._fit_global_pca()
    
    def load_time_series(self, 
                         df: pd.DataFrame, 
                         date_col: str, 
                         ticker_col: str, 
                         feature_cols: List[str], 
                         freq: str = 'M',
                         missing_strategy: str = 'mean'):
        """
        Deprecated: Use load_data() instead.
        
        This method is kept for backward compatibility.
        """
        import warnings
        warnings.warn(
            "load_time_series() is deprecated and will be removed in a future version. "
            "Use load_data() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.load_data(df, ticker_col, feature_cols, date_col, freq, missing_strategy)

    def get_dates(self) -> np.ndarray:
        if self.df is None: return np.array([])
        return self.df[self._date_col].unique()

    def _compute_global_color_bounds(self):
        """Pre-compute global min/max for each feature column across all timestamps."""
        if self.df is None or not self._feature_cols:
            return
        self._global_color_bounds = {}
        for col in self._feature_cols:
            vals = self.df[col].values
            self._global_color_bounds[col] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
        print(f"Computed global color bounds for {len(self._feature_cols)} features.")
    
    def _fit_global_pca(self, color_feature: Union[int, str, None] = None):
        """
        Pre-computes a single global PCA fit using ALL features.
        """
        if self.df is None or not self._feature_cols:
            return
        
        cache_key = "global"
        if cache_key in self._global_pca_cache:
            return
            
        print(f"Fitting global PCA on all {len(self._feature_cols)} features...")
        X_all = self.df[self._feature_cols].values
        pca_info = self.processor.fit_global_pca(X_all, n_components=3, 
                                                  feature_names=self._feature_cols)
        
        # Pre-compute global normalization bounds by projecting all timestamps
        dates = self.get_dates()
        all_positions = []
        for date in dates:
            mask = self.df[self._date_col] == date
            snapshot_X = self.df.loc[mask, self._feature_cols].values
            positions = pca_info['pca'].transform(snapshot_X)
            all_positions.append(positions)
        
        global_center, global_scale = self.processor.compute_global_normalization_bounds(all_positions)
        
        self._global_pca_cache[cache_key] = {
            **pca_info,
            'feature_names': self._feature_cols,
            'global_center': global_center,
            'global_scale': global_scale
        }
        print(f"Global PCA fit complete. Cached as '{cache_key}'.")
    
    def _get_global_pca_for_color(self, color_feature: Union[int, str, None] = None) -> Optional[Dict[str, Any]]:
        """Get cached global PCA info for all features, fitting if needed."""
        if "global" not in self._global_pca_cache:
            self._fit_global_pca()
        
        return self._global_pca_cache.get("global")

    def prepare_frame(self, date: Union[str, pd.Timestamp], method: str = 'pca', 
                       n_components: int = 3, color_feature: Union[int, str, None] = None,
                       normalization: str = 'per_frame') -> Dict[str, Any]:
        """
        Detailed pipeline for a single date:
        1. Filter DF
        2. Extract features
        3. Dim Reduction
        4. Normalize
        5. Return positions (Nx3), values (Nx1), tickers (N), color_label
        
        Args:
            color_feature: None = auto (highest variance), int = column index, str = column name
            normalization: 'global' = use global PCA fit and global bounds for temporal
                           continuity across frames. 'per_frame' = original per-frame 
                           fit behavior (default, backward compatible).
        """
        if self.df is None: return {}
        
        # 1. Filter
        mask = self.df[self._date_col] == pd.to_datetime(date)
        snapshot = self.df.loc[mask].copy()
        
        if snapshot.empty:
            return {}
            
        # 2. Extract Features
        X = snapshot[self._feature_cols].values # (N, F)
        
        # 3. Select Color Feature
        if color_feature is None:
            variances = np.var(X, axis=0)
            color_idx = int(np.argmax(variances))
        elif isinstance(color_feature, str):
            if color_feature in self._feature_cols:
                color_idx = self._feature_cols.index(color_feature)
            else:
                color_idx = 0
        else:
            color_idx = int(color_feature) % len(self._feature_cols)
            
        color_label = self._feature_cols[color_idx]
        
        # Extract color values with global or per-frame normalization
        color_col_name = self._feature_cols[color_idx]
        color_values = X[:, color_idx].copy()
        
        if normalization == 'global' and color_col_name in self._global_color_bounds:
            # Global: use pre-computed min/max across ALL timestamps
            v_min, v_max = self._global_color_bounds[color_col_name]
        else:
            # Per-frame: use this frame's min/max (original behavior)
            v_min, v_max = color_values.min(), color_values.max()
        
        if v_max > v_min:
            color_values = (color_values - v_min) / (v_max - v_min)
            color_values = np.clip(color_values, 0.0, 1.0)
        else:
            color_values = np.zeros_like(color_values)
        
        X_for_pca = X
        feature_names_for_pca = self._feature_cols
        
        # 5. Dim Reduction — global or per-frame
        global_cache = None
        if normalization == 'global' and method == 'pca':
            global_cache = self._get_global_pca_for_color(color_feature)
        
        if global_cache is not None:
            # Use global PCA: transform only, consistent axes across frames
            reduction_result = self.processor.reduce_dimensions_with_info(
                X_for_pca, method=method, n_components=n_components, 
                feature_names=feature_names_for_pca,
                fitted_pca=global_cache['pca']
            )
        else:
            # Per-frame PCA fit (original behavior)
            reduction_result = self.processor.reduce_dimensions_with_info(
                X_for_pca, method=method, n_components=n_components, 
                feature_names=feature_names_for_pca
            )
        positions_raw = reduction_result['positions']
        
        # 6. Normalize — global or per-frame
        if global_cache is not None:
            positions_norm = self.processor.normalize_positions(
                positions_raw, scale=10.0,
                global_center=global_cache['global_center'],
                global_scale=global_cache['global_scale']
            )
        else:
            positions_norm = self.processor.normalize_positions(positions_raw, scale=10.0)
        
        # 7. Generate axis labels
        axis_labels = self._generate_axis_labels(
            method, 
            reduction_result.get('explained_variance_ratios'),
            reduction_result.get('top_features_per_axis'),
            reduction_result.get('top_loadings_per_axis')
        )
        
        tickers = snapshot[self._ticker_col].values
        
        # 8. Compute per-feature statistics (Phase 1: Stats Panel)
        stats = []
        for i, col in enumerate(self._feature_cols):
            col_vals = X[:, i]
            stats.append({
                "name": col,
                "min": float(np.nanmin(col_vals)),
                "max": float(np.nanmax(col_vals)),
                "mean": float(np.nanmean(col_vals)),
                "std": float(np.nanstd(col_vals)),
                "median": float(np.nanmedian(col_vals)),
                "count": int(len(col_vals))
            })

        return {
            "positions": positions_norm.astype(np.float32),
            "values": color_values.astype(np.float32),
            "tickers": tickers,
            "color_label": color_label,
            "x_label": axis_labels[0],
            "y_label": axis_labels[1],
            "z_label": axis_labels[2],
            "stats": stats,
            "explained_variance": reduction_result.get('explained_variance_ratios'),
            "all_feature_values": X.astype(np.float32)  # N x F matrix for tooltips
        }
    
    def _generate_axis_labels(self, method: str, explained_var: Optional[List[float]], 
                               top_features: Optional[List[List[str]]],
                               top_loadings: Optional[List[List[float]]] = None) -> List[str]:
        """Generate human-readable axis labels."""
        labels = []
        axis_names = ['X', 'Y', 'Z']
        
        for i in range(3):
            if method == 'pca' and explained_var is not None and top_features is not None:
                var_pct = int(explained_var[i] * 100)
                
                # Format: Feature(loading), Feature(loading), ...
                if top_loadings is not None and len(top_loadings) > i:
                    feature_parts = []
                    for j in range(min(3, len(top_features[i]))):
                        fname = top_features[i][j]
                        loading = top_loadings[i][j]
                        feature_parts.append(f"{fname}({loading:.2f})")
                    feat_str = ", ".join(feature_parts)
                else:
                    # Fallback to old format if loadings not available
                    feat_str = ", ".join(top_features[i][:3])
                
                # Shorten if too long
                if len(feat_str) > 50:
                    feat_str = feat_str[:47] + "..."
                    
                labels.append(f"PC{i+1} ({var_pct}%) -> {feat_str}")
            elif method == 'tsne':
                labels.append(f"t-SNE {i+1}")
            elif method == 'umap':
                labels.append(f"UMAP {i+1}")
            else:
                labels.append(f"Axis {i+1}")
        
        return labels
        
    def animate(self, start_date: str, end_date: str, method: str = 'pca',
                normalization: str = 'global'):
        """
        Orchestrates the animation loop for time series data.
        
        Args:
            start_date: Start date for animation range.
            end_date: End date for animation range.
            method: Dimensionality reduction method ('pca', 'tsne', 'umap').
            normalization: 'global' = consistent axes/scale/color across all frames
                           (recommended for time series). 'per_frame' = independent 
                           PCA fit per frame (original behavior).
        """
        if not self.engine:
            print("Engine not initialized.")
            return

        dates = self.get_dates()
        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)
        
        valid_dates = [d for d in dates if s <= d <= e]
        
        print(f"Starting animation across {len(valid_dates)} timestamps (normalization={normalization})...")
        
        for i in range(len(valid_dates) - 1):
            t_curr = valid_dates[i]
            t_next = valid_dates[i+1]
            
            print(f"Morphing: {t_curr} -> {t_next}")
            
            # Prepare Dataframes with specified normalization mode
            data_curr = self.prepare_frame(t_curr, method=method, normalization=normalization)
            data_next = self.prepare_frame(t_next, method=method, normalization=normalization)
            
            if not data_curr or not data_next:
                continue
                
            # --- ALIGNMENT LOGIC ---
            ticks_curr = data_curr['tickers']
            ticks_next = data_next['tickers']
            
            # Intersection
            common_tickers = np.intersect1d(ticks_curr, ticks_next)
            
            if len(common_tickers) == 0:
                print("No common tickers, skipping frame.")
                continue
            
            # Filter & Align Current
            mask_curr = np.isin(ticks_curr, common_tickers)
            pos_curr_filtered = data_curr['positions'][mask_curr]
            val_curr_filtered = data_curr['values'][mask_curr]
            ticks_curr_filtered = ticks_curr[mask_curr]
            
            sort_idx_curr = np.argsort(ticks_curr_filtered)
            pos_curr_aligned = pos_curr_filtered[sort_idx_curr]
            val_curr_aligned = val_curr_filtered[sort_idx_curr]
            
            # Filter & Align Next
            mask_next = np.isin(ticks_next, common_tickers)
            pos_next_filtered = data_next['positions'][mask_next]
            val_next_filtered = data_next['values'][mask_next]
            ticks_next_filtered = ticks_next[mask_next]
            
            sort_idx_next = np.argsort(ticks_next_filtered)
            pos_next_aligned = pos_next_filtered[sort_idx_next]
            val_next_aligned = val_next_filtered[sort_idx_next]
            
            # 1. Contiguous Array oluştur
            p_c = np.ascontiguousarray(pos_curr_aligned, dtype=np.float32)
            v_c = np.ascontiguousarray(val_curr_aligned, dtype=np.float32)
            p_n = np.ascontiguousarray(pos_next_aligned, dtype=np.float32)
            v_n = np.ascontiguousarray(val_next_aligned, dtype=np.float32)
            
            # 2. Send to C++
            self.engine.set_points_raw(p_c, v_c)
            self.engine.set_target_points(p_n, v_n)
            
            # 3. Send metadata to UI (labels, stats, feature values)
            self._send_metadata_to_engine(data_curr)
            
            print(f"   -> {len(common_tickers)} points sent to GPU.")
            
            # Wait for user to view animation, polling for feature changes
            wait_time = 1.0
            steps = int(wait_time / 0.05)
            current_color_feature = color_feature
            
            for _ in range(steps):
                if hasattr(self.engine, 'is_running') and not self.engine.is_running():
                    break
                if hasattr(self.engine, 'has_color_feature_changed') and self.engine.has_color_feature_changed():
                    current_color_feature = self.engine.get_selected_color_feature_index()
                    print(f"UI changed color feature to index {current_color_feature}")
                time.sleep(0.05)
            
            # If changed, update the parameter for the next frames
            if current_color_feature != color_feature:
                color_feature = current_color_feature

    def static(self, date: Optional[str] = None, method: str = 'pca', block: bool = True):
        """
        Displays a static (non-animated) visualization for a single timestamp.
        Perfect for non-time series data or when you want to view a single snapshot.
        
        Args:
            date: Specific date to visualize. If None, uses the first available date.
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            block: If True, blocks the python execution until the visualizer window is closed.
        """
        if not self.engine:
            print("Engine not initialized.")
            return
        
        # Get available dates
        dates = self.get_dates()
        
        if len(dates) == 0:
            print("No data loaded. Use load_time_series() first.")
            return
        
        # Select date
        if date is None:
            selected_date = dates[0]
            print(f"No date specified, using first available: {selected_date}")
        else:
            selected_date = pd.to_datetime(date)
            if selected_date not in dates:
                print(f"Warning: Date {date} not found. Available dates: {dates[:5]}...")
                print(f"Using closest date: {dates[0]}")
                selected_date = dates[0]
        
        # Prepare data
        print(f"Preparing static visualization for {selected_date}...")
        data = self.prepare_frame(selected_date, method=method)
        
        if not data:
            print("Failed to prepare data for the selected date.")
            return
        
        # Convert to contiguous arrays for C++
        positions = np.ascontiguousarray(data['positions'], dtype=np.float32)
        values = np.ascontiguousarray(data['values'], dtype=np.float32)
        
        # Send to engine (set both current and target to same data for static view)
        self.engine.set_points_raw(positions, values)
        self.engine.set_target_points(positions, values)
        
        # Send all metadata to engine (labels, stats, feature values)
        self._send_metadata_to_engine(data)
        
        print(f"✓ Loaded {len(positions)} points to GPU.")
        print(f"  Color: {data.get('color_label', 'N/A')}")
        print(f"  X-axis: {data.get('x_label', 'N/A')}")
        print(f"  Y-axis: {data.get('y_label', 'N/A')}")
        print(f"  Z-axis: {data.get('z_label', 'N/A')}")
        print("\nVisualization ready. Interact with the 3D view!")
        
        if block:
            try:
                # Wait until the user closes the window (m_running becomes false)
                current_color_feature = None
                while hasattr(self.engine, 'is_running') and self.engine.is_running():
                    if hasattr(self.engine, 'has_color_feature_changed') and self.engine.has_color_feature_changed():
                        current_color_feature = self.engine.get_selected_color_feature_index()
                        print(f"UI requested feature change to index {current_color_feature}")
                        
                        # Update data with new color feature
                        new_data = self.prepare_frame(selected_date, method=method, color_feature=current_color_feature)
                        if new_data:
                            new_positions = np.ascontiguousarray(new_data['positions'], dtype=np.float32)
                            new_values = np.ascontiguousarray(new_data['values'], dtype=np.float32)
                            self.engine.set_points_raw(new_positions, new_values)
                            self.engine.set_target_points(new_positions, new_values)
                            self._send_metadata_to_engine(new_data)
                            
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("Interrupted by user.")
                self.stop()
    
    def _send_metadata_to_engine(self, data: Dict[str, Any]):
        """
        Send dimension labels, feature names, stats, explained variance,
        and all-feature-values to the C++ engine for UI display.
        """
        if not self.engine:
            return
        
        # Dimension labels
        if hasattr(self.engine, 'set_dimension_labels'):
            self.engine.set_dimension_labels(
                data.get('color_label', 'Color'),
                data.get('x_label', 'X'),
                data.get('y_label', 'Y'),
                data.get('z_label', 'Z')
            )
        
        # Feature names for color selector dropdown
        if hasattr(self.engine, 'set_feature_names') and self._feature_cols:
            self.engine.set_feature_names(self._feature_cols)
        
        # Stats panel data
        if hasattr(self.engine, 'set_stats') and 'stats' in data:
            self.engine.set_stats(data['stats'])
        
        # PCA explained variance
        if hasattr(self.engine, 'set_explained_variance') and data.get('explained_variance'):
            ev = np.array(data['explained_variance'], dtype=np.float32)
            self.engine.set_explained_variance(np.ascontiguousarray(ev))
        
        # All feature values for enhanced tooltips
        if hasattr(self.engine, 'set_all_feature_values') and 'all_feature_values' in data:
            fv = np.ascontiguousarray(data['all_feature_values'], dtype=np.float32)
            self.engine.set_all_feature_values(fv)
    
    # --- Phase 2: Selection Export ---
    
    def get_selected_points(self, date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get a DataFrame of the currently selected points (from UI rectangle or click selection).
        
        Args:
            date: Date to look up points from. If None, uses the first available date.
            
        Returns:
            DataFrame with ticker + all feature columns for selected points, or None.
        """
        if not self.engine or self.df is None:
            return None
        
        if not hasattr(self.engine, 'get_selected_ids'):
            print("Engine does not support get_selected_ids. Rebuild required.")
            return None
        
        selected_ids = self.engine.get_selected_ids()
        if not selected_ids:
            print("No points selected.")
            return None
        
        # Get the date's data
        dates = self.get_dates()
        if date is None:
            selected_date = dates[0]
        else:
            selected_date = pd.to_datetime(date)
        
        mask = self.df[self._date_col] == selected_date
        snapshot = self.df.loc[mask].copy()
        
        if snapshot.empty:
            return None
        
        # Filter by selected IDs (IDs are positional indices in the snapshot)
        valid_ids = [i for i in selected_ids if 0 <= i < len(snapshot)]
        if not valid_ids:
            print("No valid IDs in current snapshot.")
            return None
        
        result = snapshot.iloc[valid_ids][[self._ticker_col] + self._feature_cols].copy()
        result = result.reset_index(drop=True)
        
        print(f"Selected {len(result)} points.")
        return result
    
    def export_selection(self, path: str, date: Optional[str] = None) -> bool:
        """
        Export selected points to a CSV file.
        
        Args:
            path: Output CSV file path.
            date: Date to look up points from. If None, uses first available date.
            
        Returns:
            True if export was successful, False otherwise.
        """
        df = self.get_selected_points(date)
        if df is None or df.empty:
            print("Nothing to export.")
            return False
        
        df.to_csv(path, index=False)
        print(f"Exported {len(df)} points to {path}")
        return True
                
    # --- Phase 3: ML Integration ---
    
    def compute_clusters(self, n_clusters: int = 5, method: str = 'kmeans'):
        """
        Computes global clusters across all data points and adds 'Cluster' as a feature.
        
        Args:
            n_clusters: Number of clusters to find.
            method: Clustering algorithm ('kmeans').
        """
        if self.df is None or not self._feature_cols:
            print("No data loaded to compute clusters.")
            return
            
        print(f"Computing {n_clusters} clusters globally using {method}...")
        X = self.df[self._feature_cols].values
        labels = self.processor.compute_clusters(X, n_clusters=n_clusters, method=method)
        
        feature_name = 'Cluster'
        self.df[feature_name] = labels
        
        if feature_name not in self._feature_cols:
            self._feature_cols.append(feature_name)
            
        # Re-compute global bounds and PCA to include the new feature
        self._compute_global_color_bounds()
        self._global_pca_cache.clear()
        self._fit_global_pca(feature_name)
        
        print(f"✓ Added '{feature_name}' as a new feature. Select it from the UI Color Feature dropdown.")
        
    def compute_outliers(self, contamination: float = 0.05, method: str = 'isolation_forest'):
        """
        Computes global outlier scores and adds 'Outlier_Score' as a feature.
        
        Args:
            contamination: Expected proportion of outliers.
            method: Outlier detection algorithm ('isolation_forest').
        """
        if self.df is None or not self._feature_cols:
            print("No data loaded to compute outliers.")
            return
            
        print(f"Detecting outliers globally using {method}...")
        X = self.df[self._feature_cols].values
        scores = self.processor.detect_outliers(X, contamination=contamination, method=method)
        
        feature_name = 'Outlier_Score'
        self.df[feature_name] = scores
        
        if feature_name not in self._feature_cols:
            self._feature_cols.append(feature_name)
            
        # Re-compute global bounds and PCA to include the new feature
        self._compute_global_color_bounds()
        self._global_pca_cache.clear()
        self._fit_global_pca(feature_name)
        
        print(f"✓ Added '{feature_name}' as a new feature. Select it from the UI Color Feature dropdown.")

    def stop(self):
        if self.engine:
            self.engine.stop()
