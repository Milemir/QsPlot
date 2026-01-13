import numpy as np
import pandas as pd
import time
from typing import List, Optional, Union, Dict, Any
from .processor import DataProcessor

# Import the C++ Module (compiled extension)
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
        
    def load_time_series(self, 
                         df: pd.DataFrame, 
                         date_col: str, 
                         ticker_col: str, 
                         feature_cols: List[str], 
                         freq: str = 'M',
                         missing_strategy: str = 'mean'):
        """
        Loads and prepares the DataFrame for analysis.
        """
        print(f"Loading {len(df)} rows...")
        self.df = df.copy()
        
        # Convert date column
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

    def get_dates(self) -> np.ndarray:
        if self.df is None: return np.array([])
        return self.df[self._date_col].unique()

    def prepare_frame(self, date: Union[str, pd.Timestamp], method: str = 'pca', 
                       n_components: int = 3, color_feature: Union[int, str, None] = None) -> Dict[str, Any]:
        """
        Detailed pipeline for a single date:
        1. Filter DF
        2. Extract features
        3. Dim Reduction
        4. Normalize
        5. Return positions (Nx3), values (Nx1), tickers (N), color_label
        
        Args:
            color_feature: None = auto (highest variance), int = column index, str = column name
        """
        if self.df is None: return {}
        
        # 1. Filter
        mask = self.df[self._date_col] == pd.to_datetime(date)
        snapshot = self.df.loc[mask].copy()
        
        if snapshot.empty:
            return {}
            
        # 2. Extract Features
        X = snapshot[self._feature_cols].values # (N, F)
        
        # 3. Select Color Feature FIRST (before PCA)
        if color_feature is None:
            # Auto: pick feature with highest variance
            variances = np.var(X, axis=0)
            color_idx = int(np.argmax(variances))
            color_label = f"{self._feature_cols[color_idx]} (auto)"
        elif isinstance(color_feature, str):
            # By name
            if color_feature in self._feature_cols:
                color_idx = self._feature_cols.index(color_feature)
                color_label = color_feature
            else:
                color_idx = 0
                color_label = f"{self._feature_cols[0]} (fallback)"
        else:
            # By index
            color_idx = int(color_feature) % len(self._feature_cols)
            color_label = self._feature_cols[color_idx]
        
        # Extract color values
        color_values = X[:, color_idx].copy()
        v_min, v_max = color_values.min(), color_values.max()
        if v_max > v_min:
            color_values = (color_values - v_min) / (v_max - v_min)
        else:
            color_values = np.zeros_like(color_values)
        
        # 4. INDEPENDENT MODE: Exclude color feature from PCA input
        # This gives 4 independent dimensions: Color + X + Y + Z
        feature_mask = np.ones(X.shape[1], dtype=bool)
        feature_mask[color_idx] = False  # Exclude color feature
        X_for_pca = X[:, feature_mask]
        feature_names_for_pca = [f for i, f in enumerate(self._feature_cols) if i != color_idx]
        
        # 5. Dim Reduction with info for axis labels
        reduction_result = self.processor.reduce_dimensions_with_info(
            X_for_pca, method=method, n_components=n_components, 
            feature_names=feature_names_for_pca
        )
        positions_raw = reduction_result['positions']
        
        # 6. Normalize
        positions_norm = self.processor.normalize_positions(positions_raw, scale=10.0)
        
        # 7. Generate axis labels (hybrid format)
        axis_labels = self._generate_axis_labels(
            method, 
            reduction_result.get('explained_variance_ratios'),
            reduction_result.get('top_features_per_axis')
        )
        
        tickers = snapshot[self._ticker_col].values
        
        # C++ için float32 zorunluluğu
        return {
            "positions": positions_norm.astype(np.float32),
            "values": color_values.astype(np.float32),
            "tickers": tickers,
            "color_label": color_label,
            "x_label": axis_labels[0],
            "y_label": axis_labels[1],
            "z_label": axis_labels[2]
        }
    
    def _generate_axis_labels(self, method: str, explained_var: Optional[List[float]], 
                               top_features: Optional[List[List[str]]]) -> List[str]:
        """Generate human-readable axis labels."""
        labels = []
        axis_names = ['X', 'Y', 'Z']
        
        for i in range(3):
            if method == 'pca' and explained_var is not None and top_features is not None:
                # Hybrid format: PC1 (45%) → momentum, vol
                var_pct = int(explained_var[i] * 100)
                feat_str = ", ".join(top_features[i][:2])  # Top 2 features
                # Shorten feature names if too long
                if len(feat_str) > 20:
                    feat_str = feat_str[:17] + "..."
                labels.append(f"PC{i+1} ({var_pct}%) -> {feat_str}")
            elif method == 'tsne':
                labels.append(f"t-SNE {i+1}")
            elif method == 'umap':
                labels.append(f"UMAP {i+1}")
            else:
                labels.append(f"Axis {i+1}")
        
        return labels
        
    def run_morph_animation(self, start_date: str, end_date: str, method: str = 'pca'):
        """
        Orchestrates the animation loop.
        """
        if not self.engine:
            print("Engine not initialized.")
            return

        dates = self.get_dates()
        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)
        
        valid_dates = [d for d in dates if s <= d <= e]
        
        print(f"Starting animation across {len(valid_dates)} timestamps...")
        
        for i in range(len(valid_dates) - 1):
            t_curr = valid_dates[i]
            t_next = valid_dates[i+1]
            
            print(f"Morphing: {t_curr} -> {t_next}")
            
            # Prepare Dataframes
            data_curr = self.prepare_frame(t_curr, method=method)
            data_next = self.prepare_frame(t_next, method=method)
            
            if not data_curr or not data_next:
                continue
                
            # --- ALIGNMENT LOGIC (Bu kısım harika!) ---
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
            
            # --- SEND TO ENGINE & MEMORY SAFETY ---
            
            # 1. Contiguous Array oluştur (C++ pointer erişimi için şart)
            p_c = np.ascontiguousarray(pos_curr_aligned, dtype=np.float32)
            v_c = np.ascontiguousarray(val_curr_aligned, dtype=np.float32)
            p_n = np.ascontiguousarray(pos_next_aligned, dtype=np.float32)
            v_n = np.ascontiguousarray(val_next_aligned, dtype=np.float32)
            
            # 2. Send to C++ (C++ now copies data, no need to keep Python refs alive)
            self.engine.set_points_raw(p_c, v_c)
            self.engine.set_target_points(p_n, v_n)
            
            # 3. Send dimension labels to UI
            if hasattr(self.engine, 'set_dimension_labels'):
                self.engine.set_dimension_labels(
                    data_curr.get('color_label', 'Color'),
                    data_curr.get('x_label', 'X'),
                    data_curr.get('y_label', 'Y'),
                    data_curr.get('z_label', 'Z')
                )
            
            print(f"   -> {len(common_tickers)} points sent to GPU.")
            
            # Wait for user to view animation
            time.sleep(1.0) 
    
    def run_static_visualization(self, date: Optional[str] = None, method: str = 'pca'):
        """
        Displays a static (non-animated) visualization for a single timestamp.
        Perfect for non-time series data or when you want to view a single snapshot.
        
        Args:
            date: Specific date to visualize. If None, uses the first available date.
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
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
        
        # Send dimension labels
        if hasattr(self.engine, 'set_dimension_labels'):
            self.engine.set_dimension_labels(
                data.get('color_label', 'Color'),
                data.get('x_label', 'X'),
                data.get('y_label', 'Y'),
                data.get('z_label', 'Z')
            )
        
        print(f"✓ Loaded {len(positions)} points to GPU.")
        print(f"  Color: {data.get('color_label', 'N/A')}")
        print(f"  X-axis: {data.get('x_label', 'N/A')}")
        print(f"  Y-axis: {data.get('y_label', 'N/A')}")
        print(f"  Z-axis: {data.get('z_label', 'N/A')}")
        print("\nVisualization ready. Interact with the 3D view!")
                
    def stop(self):
        if self.engine:
            self.engine.stop()