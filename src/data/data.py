import polars as pl
import yfinance as yf
import csv
from pathlib import Path
from .features import Features


# Lazy import for torch to avoid DLL issues when not needed
try:
    from torch.utils.data import Dataset
except (ImportError, OSError):
    Dataset = object  # Fallback for when torch is not available


class DataExtractor:
    """Extracts data from various sources (yfinance, CSV, macro data, etc.)"""

    def __init__(self, tickers=None):
        self.tickers = tickers

    def extract_csv(self, file_path: str):
        """Extract ticker data from CSV file. Expects 1st column to be tickers"""
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip
            self.tickers = [row[0].strip() for row in reader if row]

        return self.tickers
           

    def extract_yfinance(self, tickers: list[str]=None, start=None, end=None, period=None, interval="1mo"):
        """
        Extract close prices from Yahoo Finance

        Returns:
            DataFrame with shape (T x N) where T=time periods, N=num assets
        """
        # yf.Tickers expects a space-separated string of tickers "AAPL MSFT GOOG"
        if not tickers:
            tickers = self.tickers

            if not tickers:
                print("No tickers provided. Load from csv or manually enter a list of yfinance-compatible tickers")
                return

        tickers_str = " ".join(tickers)
        assets = yf.Tickers(tickers_str)
        assets_df = assets.download(
            period=period, interval=interval, start=start, end=end)


        # Extract only Close prices into T x N dataframe (T=time periods, N=assets)
        if 'Close' in assets_df.columns:
            close_df = assets_df[['Close']]
            close_df.columns = tickers
        else:
            close_df = assets_df['Close']

        return close_df

    def extract_macro(self):
        """Extract macro economic data"""
        # TODO: FRED. Future implementation idea - pull from DB?
        pass


class DataProcessor:
    """
    Streamlines data processing, cleaning, and feature engineering.
    
    Uses a builder/chaining pattern for easy composition of data processing steps.
    
    Example usage:
        processor = DataProcessor(timeseries, date_col="Date", null_threshold=0.1)
        processor.clean_data().compute_returns().add_momentum(1).add_momentum(12).add_volatility(3).add_volatility(12).finalize()
        X, Y = processor.get_features_and_returns()
    """
    
    def __init__(self, timeseries_data, date_col: str = "Date", null_threshold: float = 0.1):
        """
        Initialize the data processor with raw timeseries data.
        
        Args:
            timeseries_data: Pandas DataFrame with timeseries data (Date index + asset columns)
            date_col: Name of the date column (default: "Date")
            null_threshold: Fraction of missing data allowed per column (default: 0.1 = 10%)
        """
        # Convert to polars if needed
        if hasattr(timeseries_data, 'reset_index'):
            # It's a pandas DataFrame - polars doesn't have 'reset_index'
            self.data = pl.DataFrame(timeseries_data.reset_index())
        else:
            self.data = timeseries_data
        
        self.date_col = date_col
        self.null_threshold = null_threshold
        
        self.asset_cols = [col for col in self.data.columns if col != date_col]
        self.returns_pl = None

        # combined - df with features columns combined
        self.combined = None
        self.feature_cols = []
        self._is_finalized = False
        self.features = None
        
    def clean_data(self):
        """
        Clean the timeseries data:
        - Drop columns with more than null_threshold missing data
        - Forward-fill remaining nulls
        
        Returns:
            self (for chaining)
        """
        # Drop columns with too many nulls
        null_threshold_count = len(self.data) * self.null_threshold
        cols_to_keep = [self.date_col]
        
        for col in self.asset_cols:
            null_count = self.data[col].null_count()
            if null_count <= null_threshold_count:
                cols_to_keep.append(col)
        
        self.data = self.data.select(cols_to_keep)
        self.asset_cols = [col for col in self.data.columns if col != self.date_col]
        
        # Forward-fill nulls
        self.data = self.data.with_columns([
            pl.col(col).forward_fill().alias(col) for col in self.asset_cols
        ])
        
        return self
    
    def compute_returns(self):
        """
        Calculate returns from price data.
        
        Returns:
            self (for chaining)
        """
        self.returns_pl = self.data.with_columns([
            pl.col(col).pct_change().alias(col) for col in self.asset_cols
        ])
        
        # Initialize combined with returns
        self.combined = self.returns_pl.clone()
        self.features = Features(self.returns_pl, date_col=self.date_col)
        
        return self
    
    def add_momentum(self, window: int, units: str = "m"):
        """
        Add momentum features (lagged compound returns).
        
        Args:
            window: Rolling window size
            units: Time units (d, w, m, y)
            
        Returns:
            self (for chaining)
        """
        if self.returns_pl is None or self.features is None:
            raise ValueError("Must call compute_returns() before adding features")

        mom_df = self.features.mom(window, units)
        
        self.combined = self.combined.join(mom_df, on=self.date_col, how="left")
        
        # Track feature column names
        new_feature_cols = [col for col in mom_df.columns if col != self.date_col]
        self.feature_cols.extend(new_feature_cols)
        
        return self
    
    def add_volatility(self, window: int, units: str = "m"):
        """
        Add volatility features (rolling standard deviation).
        
        Args:
            window: Rolling window size
            units: Time units (d, w, m, y)
            
        Returns:
            self (for chaining)
        """
        if self.returns_pl is None or self.features is None:
            raise ValueError("Must call compute_returns() before adding features")
        
        vol_df = self.features.volatility(window, units)
        
        self.combined = self.combined.join(vol_df, on=self.date_col, how="left")
        
        # Track feature column names
        new_feature_cols = [col for col in vol_df.columns if col != self.date_col]
        self.feature_cols.extend(new_feature_cols)
        
        return self
    
    def add_beta(self, window: int, bench: pl.DataFrame, bench_col: str = "benchmark", units: str = "m"):
        """
        Add beta features (rolling beta to benchmark).
        
        Args:
            window: Rolling window size
            bench: Benchmark returns DataFrame
            bench_col: Name of benchmark column
            units: Time units (d, w, m, y)
            
        Returns:
            self (for chaining)
        """
        if self.returns_pl is None or self.features is None:
            raise ValueError("Must call compute_returns() before adding features")

        beta_df = self.features.beta(window, bench, bench_col, units)
        
        self.combined = self.combined.join(beta_df, on=self.date_col, how="left")
        
        # Track feature column names
        new_feature_cols = [col for col in beta_df.columns if col != self.date_col]
        self.feature_cols.extend(new_feature_cols)
        
        return self
    
    def add_features(self, feature_configs: list[dict]):
        """
        Add multiple features from a configuration list.
        
        Args:
            feature_configs: List of dicts with feature specifications
                Example: [
                    {'type': 'momentum', 'window': 1},
                    {'type': 'momentum', 'window': 12},
                    {'type': 'volatility', 'window': 3},
                    {'type': 'beta', 'window': 12, 'bench': bench_df, 'bench_col': 'SPY'}
                ]
        
        Returns:
            self (for chaining)
        """
        for config in feature_configs:
            feature_type = config.get('type')
            window = config.get('window')
            units = config.get('units', 'm')
            
            if feature_type == 'momentum':
                self.add_momentum(window, units)
            elif feature_type == 'volatility':
                self.add_volatility(window, units)
            elif feature_type == 'beta':
                bench = config.get('bench')
                bench_col = config.get('bench_col', 'benchmark')
                if bench is None:
                    raise ValueError("Beta feature requires 'bench' parameter")
                self.add_beta(window, bench, bench_col, units)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
        
        return self
    
    def finalize(self, max_window: int = None):
        """
        Finalize the data processing:
        - Remove initial NaNs from rolling windows
        - Drop any remaining null rows
        
        Args:
            max_window: Maximum window size used (for removing initial NaNs)
                       If None, will be inferred from feature columns
        
        Returns:
            self (for chaining)
        """
        if self.combined is None:
            raise ValueError("No data to finalize. Did you forget to compute returns?")
        
        # Infer max_window if not provided
        if max_window is None:
            # Try to extract window sizes from feature column names
            max_window = 13  # Default conservative value
            
        # Remove initial NaNs from rolling windows
        self.combined = self.combined.slice(max_window, self.combined.height - max_window)
        
        # Drop any remaining nulls
        total_nulls = self.combined.null_count().sum_horizontal()[0]
        if total_nulls > 0:
            self.combined = self.combined.drop_nulls()
        
        self._is_finalized = True
        
        return self
    
    def get_features_and_returns(self):
        """
        Get feature matrix (X) and returns matrix (Y) in wide format.
        
        Returns:
            tuple: (X, Y) as numpy arrays
                X: shape (T, d_features) - feature matrix
                Y: shape (T, n_assets) - returns matrix
        """
        if not self._is_finalized:
            raise ValueError("Must call finalize() before extracting features and returns")
        
        X = self.combined.select(self.feature_cols).to_numpy()
        Y = self.combined.select(self.asset_cols).to_numpy()
        
        return X, Y
    
    def get_features_and_returns_stacked(self):
        """
        Get data in per-asset format for flexible model training.
        
        Each row represents one asset at one time point, enabling the model
        to learn patterns that generalize across all assets.
        
        Returns:
            tuple: (X_stacked, Y_stacked, metadata) where:
                X_stacked: numpy array of shape (T × n_assets, n_features_per_asset)
                          Each row is features for one asset at one time
                Y_stacked: numpy array of shape (T × n_assets, 1)
                          Each row is the return for one asset at one time
                metadata: dict containing:
                    - 'asset_names': array of asset names (repeated for each time)
                    - 'dates': array of dates (repeated for each asset)
                    - 'asset_indices': which asset (0 to n_assets-1)
                    - 'time_indices': which time period (0 to T-1)
                    - 'n_assets': number of unique assets
                    - 'n_times': number of time periods
                    - 'n_features_per_asset': features per asset
        
        Example:
            If you have 3 stocks (AAPL, MSFT, GOOGL) over 2 months:
            
            Wide format:
            X: [[AAPL_f1, AAPL_f2, MSFT_f1, MSFT_f2, GOOGL_f1, GOOGL_f2],  # Jan
                [AAPL_f1, AAPL_f2, MSFT_f1, MSFT_f2, GOOGL_f1, GOOGL_f2]]  # Feb
            Y: [[AAPL_r, MSFT_r, GOOGL_r],  # Jan
                [AAPL_r, MSFT_r, GOOGL_r]]  # Feb
            
            Stacked format:
            X: [[AAPL_f1, AAPL_f2],   # Jan, AAPL
                [MSFT_f1, MSFT_f2],   # Jan, MSFT
                [GOOGL_f1, GOOGL_f2], # Jan, GOOGL
                [AAPL_f1, AAPL_f2],   # Feb, AAPL
                [MSFT_f1, MSFT_f2],   # Feb, MSFT
                [GOOGL_f1, GOOGL_f2]] # Feb, GOOGL
            Y: [[AAPL_r], [MSFT_r], [GOOGL_r], [AAPL_r], [MSFT_r], [GOOGL_r]]
        """
        if not self._is_finalized:
            raise ValueError("Must call finalize() before extracting features")
        
        # Get data in wide format
        print(self.combined.shape, "="*50, "\n", self.combined)
        X_wide = self.combined.select(self.feature_cols).to_numpy()
        Y_wide = self.combined.select(self.asset_cols).to_numpy()
        dates = self.combined.select(self.date_col).to_numpy().flatten()
        
        T, n_assets = Y_wide.shape  # T time periods, n_assets stocks
        n_feature_cols = len(self.feature_cols)
        n_features_per_asset = n_feature_cols // n_assets
        
        # Validate that features divide evenly by assets
        if n_feature_cols % n_assets != 0:
            raise ValueError(
                f"Feature columns ({n_feature_cols}) don't divide evenly by assets ({n_assets}). "
                f"Expected {n_assets} * {n_features_per_asset} = {n_feature_cols}"
            )
        
        # Reshape to be one asset per row
        X_stacked = X_wide.reshape(T, n_assets, n_features_per_asset).reshape(
            T * n_assets, n_features_per_asset)

        Y_stacked = Y_wide.reshape(T * n_assets, 1)


        # Create metadata for tracking which sample corresponds to which asset/time
        import numpy as np
        
        # For each time period, list all assets

        # TO LEARN: what does this do?
        asset_names_repeated = np.tile(self.asset_cols, T)  # [A,B,C, A,B,C, ...]
        dates_repeated = np.repeat(dates, n_assets)         # [t1,t1,t1, t2,t2,t2, ...]
        
        metadata = {
            'asset_names': asset_names_repeated,
            'dates': dates_repeated,
            'asset_indices': np.tile(np.arange(n_assets), T),  # [0,1,2, 0,1,2, ...]
            'time_indices': np.repeat(np.arange(T), n_assets), # [0,0,0, 1,1,1, ...]
            'n_assets': n_assets,
            'n_times': T,
            'n_features_per_asset': n_features_per_asset,
        }
        
        return X_stacked, Y_stacked, metadata
    
    def get_dates(self):
        """Get date column as numpy array."""
        if not self._is_finalized:
            raise ValueError("Must call finalize() before extracting dates")
        
        return self.combined.select(self.date_col).to_numpy()
    
    def get_latest_features(self):
        """
        Get the most recent features for prediction.
        
        Returns:
            tuple: (latest_features, latest_date)
                latest_features: numpy array of shape (1, d_features)
                latest_date: date value
        """
        if not self._is_finalized:
            raise ValueError("Must call finalize() before extracting latest features")
        
        latest_features = self.combined.select(self.feature_cols).tail(1).to_numpy()
        latest_date = self.combined.select(self.date_col).tail(1).to_numpy()[0][0]
        
        return latest_features, latest_date
    
    def get_asset_columns(self):
        """Get list of asset column names."""
        return self.asset_cols
    
    def get_feature_columns(self):
        """Get list of feature column names."""
        return self.feature_cols
    
    def get_combined_dataframe(self):
        """Get the full combined DataFrame (returns + features)."""
        if not self._is_finalized:
            raise ValueError("Must call finalize() before accessing combined dataframe")
        
        return self.combined

class CustomDataset:
    """Manages and stores extracted market data"""

    def __init__(self, data: pl.DataFrame = None, data_dir: str = None, data_file: str = None, *bench_data_files: str):
        """
        Initialize dataset with optional CSV files

        Args:
            data_dir: Directory containing data files
            data_file: Main data file name
            bench_data_files: Benchmark data file names
        """
        # TODO: do we need to classify different datasets? Incl. a name attribute?
        self.data = data
        self.bench_data = {}

        # Load from CSV if paths provided
        if data_dir and data_file:
            self.data = pl.read_csv(Path(data_dir) / data_file)
            self.bench_data = {
                bench: pl.read_csv(Path(data_dir) / bench)
                for bench in bench_data_files
            }
        else:
            # else, initialize returns (computed lazily when needed)
            self.set_data(None)

    def _compute_returns(self):
        """Compute returns from price data"""
        if self.data is None:
            self._returns = None
            return

        # TODO: handle NaNs, synchronise trading days
        self.returns = self.data.with_columns(
            # assumes the Date column is the first
            pl.col(col).pct_change() for col in self.assets[1:]
        )

    def set_data(self, data):
        """Store extracted data and recompute returns"""
        self.data = data
        self._compute_returns()
        return self

    def get_data(self):
        """Get stored price data"""
        return self.data

    def get_returns(self):
        """Get returns matrix (T x N)"""
        return self._returns

    def get_bench_data(self, bench: str):
        """Get benchmark data by name"""
        return self.bench_data.get(bench)


class CostDataset(Dataset):
    """For internal usage; DataLoader wraps me, used for readability, clean code and also multithreading is easy

    Covariance modes:
    - 'static': Single covariance matrix for all samples (simple, stable)
    - 'rolling': Per-sample covariance from rolling window (realistic, no look-ahead bias)
    """

    def __init__(self, X, Y):
        """
        Args:
            X: Features tensor, shape (T, d_features)
            Y: Returns tensor, shape (T, n_assets), where each row contains expected returns for each asset
        """
        self.X = X  # shape = (T, d_features)
        self.Y = Y  # shape = (T, n_assets) -> returns


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
