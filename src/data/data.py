import polars as pl
import yfinance as yf
from pathlib import Path

# Lazy import for torch to avoid DLL issues when not needed
try:
    from torch.utils.data import Dataset
except (ImportError, OSError):
    Dataset = object  # Fallback for when torch is not available


class DataExtractor:
    """Extracts data from various sources (yfinance, CSV, macro data, etc.)"""

    def extract_csv(self, file_path: str):
        """Extract data from CSV file"""
        # TODO: implementation
        pass

    def extract_yfinance(self, tickers: list[str], start=None, end=None, period=None, interval="1mo"):
        """
        Extract close prices from Yahoo Finance

        Returns:
            DataFrame with shape (T x N) where T=time periods, N=num assets
        """
        # yf.Tickers expects a space-separated string of tickers "AAPL MSFT GOOG"
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

    def __init__(self, X, C, cov, rf):
        """
        Args:
            X: Features tensor, shape (T, d_features)
            C: Cost vectors tensor, shape (T, n_assets), where costs = -expected_returns
            cov: Either:
                - Single covariance matrix (n_assets, n_assets) for static mode
                - Stacked covariance matrices (T, n_assets, n_assets) for rolling mode
            rf: Risk-free rate (scalar)
        """
        self.X = X  # shape = (T, d_features)
        self.C = C  # shape = (T, n_assets) -> costs (negative returns)
        self.rf = rf

        # Determine covariance mode based on shape
        if cov.dim() == 2:
            # Static mode: single (n_assets, n_assets) matrix
            self.cov_mode = 'static'
            self.cov = cov
        elif cov.dim() == 3:
            # Rolling mode: (T, n_assets, n_assets) stacked matrices
            self.cov_mode = 'rolling'
            self.cov = cov
            assert cov.shape[0] == len(X), \
                f"Rolling cov samples ({cov.shape[0]}) must match data samples ({len(X)})"
        else:
            raise ValueError(
                f"cov must be 2D (static) or 3D (rolling), got {cov.dim()}D")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.cov_mode == 'static':
            return self.X[idx], self.C[idx], self.cov, self.rf
        else:
            # Return the covariance matrix for this specific time point
            return self.X[idx], self.C[idx], self.cov[idx], self.rf
