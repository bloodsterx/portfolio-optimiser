from torch.utils.data import Dataset
import polars as pl
from pathlib import Path

class CustomDataset():
    def __init__(self, data_dir: str, data_file: str, *bench_data_files: list[str]):
        self.data = pl.read_csv(Path.joinpath(data_dir, data_file))
        self.bench_data_files = {bench: pl.read_csv(Path.joinpath(data_dir, bench)) for bench in bench_data_files}

    def get_data(self):
        return self.data

    def get_bench_data(self, bench):
        return self.bench_data_files.get(bench)
    



class CostDataset(Dataset):
    # for internal usage; DataLoader wraps me, used for readability, clean code and also multithreading is easy

    def __init__(self, X, C):
        self.X = X # shape = (T-1, d_features) for T-1 time period
        self.C = C # shape = (T-1, n_assets) -> actual returns 'costs'

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # retrieve one data sample
        return self.X[idx], self.C[idx]