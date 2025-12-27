from typing import Any


from collections import defaultdict
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

        epochs = []
        train_losses = []
        val_losses = []
        avg_train_losses = []
        avg_val_losses = []

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

                    val_loss += loss.item() * X_batch.size(0)
                    n_val_samples += X_batch.size(0)



            avg_val_loss = val_loss / n_val_samples

            if epoch % 5 == 0:
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                avg_train_losses.append(avg_train_loss)
                avg_val_losses.append(avg_val_loss)
                
                print(f"Epoch {epoch:3d} | "
                    f"train_loss: {avg_train_loss:.4f} | "
                    f"val_loss: {avg_val_loss:.4f} | ")
            
        
        output = {
            "epochs": epochs,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "avg_train_losses": avg_train_losses,
            "avg_val_losses": avg_val_losses,
        }

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


def run(
    device="cpu", 
    tickers=None, 
    data_path=None,
    data_period="5y",
    interval="1mo"
    ):

    # <== EXTRACT DATA ==>

    extractor = DataExtractor()
    if not tickers:
        extractor.extract_csv(data_path)
    
    timeseries = extractor.extract_yfinance(
        period=data_period,     
        interval=interval
    )

    print(f"Loaded {len(extractor.tickers)} tickers: {extractor.tickers[:10]}...")  # Preview first 10
    
    # <== DATA PROCESSING & CLEANING ==>

    timeseries_pl = pl.DataFrame(timeseries.reset_index()) # reset index to be the Date column
    asset_cols = [col for col in timeseries_pl.columns if col != "Date"]
    
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
    
    # Handling Nulls - Forward-fill
    timeseries_pl = timeseries_pl.with_columns([
        pl.col(col).forward_fill().alias(col) for col in asset_cols
    ])
    
    print(f"Timeseries nulls after fill: {timeseries_pl.null_count().sum_horizontal()[0]}")

    returns_pl = timeseries_pl.with_columns([
        pl.col(col).pct_change().alias(col) for col in asset_cols
    ])
    
    print(f"Final shape: {returns_pl.shape} with {len(asset_cols)} assets")

    # <== FEATURE ENGINEERING (HARDCODED) ==> 
    # ATM: hardcoded, will use args later
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
    
    # Remove initial NaNs from rolling windows
    max_window = 13
    combined = combined.slice(max_window, combined.height - max_window)
    # Check if there are any remaining nulls (there shouldn't be after forward fill)
    total_nulls = combined.null_count().sum_horizontal()[0]
    if total_nulls > 0:
        print(f"Warning: Found {total_nulls} nulls after filling, dropping rows with nulls")
        combined = combined.drop_nulls()
    
    print(f"Combined features shape: {combined.shape}")
    print(f"Columns: {combined.columns[:10]}...")
    
    feature_cols = [col for col in combined.columns 
                    if col not in ["Date"] + asset_cols]

    dates = combined.select("Date").to_numpy()

    # <== DATA SPLITTING FOR TRAIN/VAL/TEST ==>
    train_dates, val_dates, test_dates = Trainer.split_train_data(dates, 0.7)

    X = combined.select(feature_cols).to_numpy()
    returns = combined.select(asset_cols).to_numpy()
    Y = returns 
    
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    X_train, X_val, X_test = Trainer.split_train_data(X, 0.7)
    Y_train, Y_val, Y_test = Trainer.split_train_data(Y, 0.7)

    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val: X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test: X={X_test.shape}, Y={Y_test.shape}")
    
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

    # <== TRAIN & EVAL ==>

    model = MLPModel(X_train_t.shape[1], len(asset_cols))
    trainer = Trainer(model, device="cuda")
    model, output = trainer.train(train_loader, val_loader, n_epochs=500)

    plt.figure(figsize=(12,6))
    plt.plot(output["epochs"], output["val_losses"], label="val_loss")
    plt.plot(output["epochs"], output["train_losses"], label="train_loss")
    plt.title("Train & Val loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig("train_val_loss.png")

    now = datetime.now()
    torch.save(model.state_dict(), f"{now.strftime('%Y-%m-%d')}-DL-weights.pt")

    model.eval()
    loss_fn = torch.nn.MSELoss()
    with torch.inference_mode():
        Y_hat = model(X_test_t)
        MSE = loss_fn(Y_hat, Y_test_t)

    print(f"MSE: {MSE}")


    
    
if __name__ == "__main__":
    device = "cuda"

    if not torch.cuda.is_available():
        device = "cpu"

    run(device=device, data_path="sp500-stocks.csv")
    


