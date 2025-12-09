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
        assets_df = assets.download(period=period, interval=interval, start=start, end=end)
        
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
            pl.col(col).pct_change() for col in self.assets[1:] # assumes the Date column is the first
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
    # for internal usage; DataLoader wraps me, used for readability, clean code and also multithreading is easy

    def __init__(self, X, C, cov, rf):
        self.X = X # shape = (T-1, d_features) for T-1 time period
        self.C = C # shape = (T-1, n_assets) -> actual returns 'costs'
        self.cov = cov
        self.rf = rf

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # retrieve one data sample
        return self.X[idx], self.C[idx], self.cov, self.rf