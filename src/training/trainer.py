from datetime import datetime
import torch.nn as nn
import torch
import numpy as np
import polars as pl
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from .model import MLPModel
from ..data.data import DataExtractor, CostDataset
from ..data.features import Features
from .oracle import MVO


class Trainer:
    
    def __init__(self, model: nn.Module, device="cpu"):
        self.model = model
        self.device = device
        
        if self.device == "cuda" and not torch.cuda.is_available(): 
            print("CUDA is not available on this machine")
            device = "cpu"

        self.model.to(device)

    def train(
        self,
        train_dataloader,
        val_dataloader,    
        optim="adam", 
        n_epochs=100, 
        lr = 1e-3, 
    ):

        match optim:
            case "adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            case "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        loss_fn = torch.nn.MSELoss()

        output = []

        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.0
            n_train_samples = 0

            for X_batch, Y_batch in train_dataloader:
                # 1. forward pass
                Y_hat_train = self.model(X_batch)

                # 2. loss calc
                loss = loss_fn(Y_hat_train, Y_batch)

                # 3. clear old gradients
                optimizer.zero_grad()  

                # 4. backprop
                loss.backward()
                
                # 5. update Params
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                n_train_samples += X_batch.size(0)
            
            avg_train_loss = train_loss / n_train_samples

            self.model.eval()
            val_loss = 0.0
            n_val_samples = 0

            # validation 
            with torch.inference_mode():
                for X_batch, Y_batch in val_dataloader:
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)

                    Y_hat_val = self.model(X_batch)
                    loss = loss_fn(Y_hat_val, Y_batch)

                    val_loss += loss
                    val_loss = val_loss.item()
                    n_val_samples += X_batch.size(0)

            avg_val_loss = val_loss / n_val_samples

            if epoch % 5 == 0:
                output.append({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                })

                print(f"Epoch {epoch:3d} | "
                    f"train_loss: {avg_train_loss:.4f} | "
                    f"val_loss: {avg_val_loss:.4f} | ")


        return self.model, output

    @staticmethod
    def split_train_data(data: np.ndarray, split: float) -> tuple[np.ndarray]:
        """Splits time series matrix into train, test and validation sets
        
        Uses chronological splitting (no random shuffling) to maintain time series dependencies.

        Args:
            data (np.ndarray): Time series data matrix (T x N) where T = time periods, N = n_assets
            split (float): Training data ratio. The remaining data is split equally between val and test.

        Returns:
            tuple[np.ndarray]: (train_data, val_data, test_data)
        """
        if not 0 < split < 1:
            raise ValueError(f"Split ratio must be between 0 and 1, got {split}")
        
        n_samples = len(data)
        
        # num. training samples
        train_partition = int(n_samples * split)
        
        # Remaining samples equally split amongst val and test (ie; for all data @ index >= train_partition)
        remaining = n_samples - train_partition
        val_size = remaining // 2
        val_partition = train_partition + val_size
        
        train_data = data[:train_partition]
        val_data = data[train_partition:val_partition]
        test_data = data[val_partition:]
        
        return train_data, val_data, test_data


def run(device="cpu"):
    # step 1. extract the features from the data
    import csv
    
    with open("sp500-stocks.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        tickers = [row[0].strip() for row in reader if row]
    
    print(f"Loaded {len(tickers)} tickers: {tickers[:10]}...")  # Preview first 10
    
    extractor = DataExtractor()
    timeseries = extractor.extract_yfinance(
        tickers=tickers,
        period="5y",     
        interval="1mo"
    )
    
    # reset index column to Date 
    timeseries_reset = timeseries.reset_index()

    # Convert to polars df
    timeseries_pl = pl.DataFrame(timeseries_reset)
    
    # Is this needed? Rename index column to Date if it isn't already
    if "Date" not in timeseries_pl.columns:
        # The index column might be named "index" or the datetime column name
        date_col_name = timeseries_pl.columns[0]  # First column should be the date
        timeseries_pl = timeseries_pl.rename({date_col_name: "Date"})
    
    # Filter out assets with too many missing values
    asset_cols = [col for col in timeseries_pl.columns if col != "Date"]
    
    print(f"Timeseries shape before cleaning: {timeseries_pl.shape}")
    print(f"Timeseries nulls before cleaning: {timeseries_pl.null_count().sum_horizontal()[0]}")
    
    # Drop columns where more than 10% of data is missing
    null_threshold = len(timeseries_pl) * 0.1
    cols_to_keep = ["Date"]
    for col in asset_cols:
        null_count = timeseries_pl[col].null_count()
        if null_count <= null_threshold:
            cols_to_keep.append(col)
    
    timeseries_pl = timeseries_pl.select(cols_to_keep)
    asset_cols = [col for col in timeseries_pl.columns if col != "Date"]
    print(f"Kept {len(asset_cols)} assets after filtering (removed {len([col for col in asset_cols if col not in cols_to_keep])} assets with > 10% missing data)")
    
    # Fill remaining missing values
    # Forward-fill: carry last known price forward, then backfill for any nulls at the start
    timeseries_pl = timeseries_pl.with_columns([
        pl.col(col).forward_fill().backward_fill().alias(col) for col in asset_cols
    ])
    
    print(f"Timeseries nulls after fill: {timeseries_pl.null_count().sum_horizontal()[0]}")
    
    # prices -> returns conversion
    returns_pl = timeseries_pl.with_columns([
        pl.col(col).pct_change().alias(col) for col in asset_cols
    ])
    
    print(f"Final shape: {returns_pl.shape} with {len(asset_cols)} assets")

    # Compute features and store in polars dataframes
    features = Features(returns_pl)
 
    mom1m = features.mom(1)
    mom12m = features.mom(12)

    # ==============================
    # skip beta for now
    # rolling_beta1m = features.beta(1, bench_data)
    # rolling_beta12m = features.beta(12, bench_data)
    # ==============================

    volatility1m = features.volatility(3)
    volatility12m = features.volatility(12)
    
    # Combine all features into a single DataFrame
    combined = returns_pl.clone()
    for feature_df in [mom1m, mom12m, volatility1m, volatility12m]:
        combined = combined.join(feature_df, on="Date", how="left")
    
    # Slice after the max rolling window to remove initial NaNs from rolling windows
    # The 12-month rolling windows create nulls for the first 12 rows + 1 for pct_change
    max_window = 13
    combined = combined.slice(max_window, combined.height - max_window)
    
    # Check if there are any remaining nulls (there shouldn't be after forward/backward fill)
    total_nulls = combined.null_count().sum_horizontal()[0]
    if total_nulls > 0:
        print(f"Warning: Found {total_nulls} nulls after filling, dropping rows with nulls")
        combined = combined.drop_nulls()
    
    print(f"Combined features shape: {combined.shape}")
    print(f"Columns: {combined.columns[:10]}...")
    
    feature_cols = [col for col in combined.columns 
                    if col not in ["Date"] + asset_cols]

    dates = combined.select("Date").to_numpy()
    train_dates, val_dates, test_dates = Trainer.split_train_data(dates, 0.7)

    X = combined.select(feature_cols).to_numpy()
    # Convert returns to costs (c = -returns) to match SPO+ paper formulation
    returns = combined.select(asset_cols).to_numpy()
    C = -returns  # Cost formulation: minimize costs = maximize returns
    
    print(f"X shape: {X.shape}, C shape: {C.shape}")

    # step 2. prepare data into splits, into datasets and create loader objects 
    X_train, X_val, X_test = Trainer.split_train_data(X, 0.7)
    Y_train, Y_val, Y_test = Trainer.split_train_data(C, 0.7)

    
    
    
    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val: X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test: X={X_test.shape}, Y={Y_test.shape}")

    device = "cuda"
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)

    # setup dataloader objects
    train_dataset = CostDataset(X_train_t, Y_train_t)
    val_dataset = CostDataset(X_val_t, Y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Created dataloaders with {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # step 3. train!
    model = MLPModel(X_train_t.shape[1], len(asset_cols))
    trainer = Trainer(model, device="cuda")
    model, output = trainer.train(train_loader, val_loader, n_epochs=500)
    now = datetime.now()
    torch.save(model.state_dict(), f"{now.strftime('%Y-%m-%d')}-DL-weights.pt")

    model.eval()
    loss_fn = torch.nn.MSELoss()
    with torch.inference_mode():
        Y_hat = model(X_test_t)
        test_loss = loss_fn(Y_hat, Y_test_t)



    
    
if __name__ == "__main__":
    run()
    


