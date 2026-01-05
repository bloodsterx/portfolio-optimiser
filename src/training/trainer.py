from datetime import datetime
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import os
import json
from .model import MLPModel
from ..data.data import DataExtractor, CostDataset, DataProcessor

SAVE_DIR = "models"

class Log:
    def __init__(
        self,
        date,
        in_features, 
        data_period,
        interval, 
        hyperparams,
        test_loss,
        feature_configs,
    ):
        self.date = date
        self.in_features = in_features
        self.data_period = data_period
        self.interval = interval
        self.hyperparams = hyperparams # batch size, n_epochs, lr, etc
        self.test_loss = test_loss
        self.feature_configs = feature_configs


    def save_log(self, save_dir):
        log_dict = {
            f"date": self.date,
            f"data_period": self.data_period,
            f"interval": self.interval,
            f"in_features": self.in_features,
            f"hyperparams": self.hyperparams,
            f"test_loss": self.test_loss,
            f"feature_configs": self.feature_configs,
        }
        with open(os.path.join(save_dir, "log.json"), "w") as f:
            json.dump(obj=log_dict, fp=f, indent=4)

class Trainer:
    
    def __init__(self, model: nn.Module, device="cpu"):
        self.model = model
        self.device = device
        
        # Safely check CUDA availability
        if self.device == "cuda":
            try:
                if not torch.cuda.is_available():
                    print("CUDA is not available on this machine")
                    self.device = "cpu"
            except (AssertionError, AttributeError):
                print("Torch not compiled with CUDA support")
                self.device = "cpu"

        self.model.to(self.device)

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
            case "sgd":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
            case _:
                raise ValueError(f"Unknown optimizer: {optim}. Supported: 'adam', 'sgd'")

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
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
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
            min_val_loss = float('inf')
            best_model_weights = self.model.state_dict()
            # validation 
            with torch.inference_mode():
                for X_batch, Y_batch in val_dataloader:
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)

                    Y_hat_val = self.model(X_batch)
                    loss = loss_fn(Y_hat_val, Y_batch)

                    val_loss += loss.item() * X_batch.size(0)
                    n_val_samples += X_batch.size(0)

                    if val_loss < min_val_loss:
                        best_model_weights = self.model.state_dict()
                        min_val_loss = val_loss

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
            "best_model_weights": best_model_weights # state dict producing the smallest validation loss
        }

        return self.model, output


def split_train_data(data: np.ndarray, split: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits time series matrix into train, validation, and test sets.
    
    Uses chronological splitting (no random shuffling) to maintain time series dependencies.

    Args:
        data: Time series data matrix (T x N) where T = time periods, N = n_assets
        split: Training data ratio. The remaining data is split equally between val and test.

    Returns:
        (train_data, val_data, test_data)
    """
    if not 0 < split < 1:
        raise ValueError(f"Split ratio must be between 0 and 1, got {split}")
    
    n_samples = len(data)
    train_end = int(n_samples * split)
    
    remaining = n_samples - train_end
    val_end = train_end + remaining // 2
    
    return data[:train_end], data[train_end:val_end], data[val_end:]


def get_device() -> str:
    """Detect best available device (CUDA or CPU)."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except (AssertionError, AttributeError):
        pass
    return "cpu"


def prepare_data(
    data_path: str = None,
    tickers: list[str] = None,
    data_period: str = "5y",
    interval: str = "1mo",
    feature_configs: list[dict] = None,
    train_split: float = 0.7,
    device: str = "cpu",
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, int]:
    """
    Extract, process, and split data into train/val/test DataLoaders.
    
    Args:
        data_path: Path to CSV file with ticker symbols
        tickers: List of ticker symbols (alternative to data_path)
        data_period: yfinance period string (e.g., "5y", "10y")
        interval: yfinance interval string (e.g., "1mo", "1wk")
        feature_configs: List of feature configuration dicts
        train_split: Fraction of data for training (rest split between val/test)
        device: Device to place tensors on
        
    Returns:
        (train_loader, val_loader, X_test_tensor, Y_test_tensor, n_assets)
    """

    extractor = DataExtractor()
    if data_path and not tickers:
        extractor.extract_csv(data_path)
    elif tickers:
        extractor.tickers = tickers
        
    timeseries = extractor.extract_yfinance(period=data_period, interval=interval)
    print(f"Loaded {len(extractor.tickers)} tickers: {extractor.tickers[:10]}...")
    
    # Data pre-processing & cleaning
    processor = DataProcessor(timeseries, date_col="Date", null_threshold=0.1)
    processor.clean_data().compute_returns().add_features(feature_configs).finalize(max_window=13)

    breakpoint()
    
    asset_cols = processor.get_asset_columns()
    print(f"Kept {len(asset_cols)} assets after filtering")
    print(f"Combined features shape: {processor.get_combined_dataframe().shape}")
    
    # Split data
    X, Y = processor.get_features_and_returns()
    X_train, X_val, X_test = split_train_data(X, train_split)
    Y_train, Y_val, Y_test = split_train_data(Y, train_split)
    
    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val: X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test: X={X_test.shape}, Y={Y_test.shape}")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)
    
    # Create dataloaders
    train_loader = DataLoader(CostDataset(X_train_t, Y_train_t), batch_size=32, shuffle=False)
    val_loader = DataLoader(CostDataset(X_val_t, Y_val_t), batch_size=32, shuffle=False)
    
    return train_loader, val_loader, X_test_t, Y_test_t, len(asset_cols)

def run_trainer(
    data_path: str = None,
    tickers: list[str] = None,
    data_period: str = "5y",
    interval: str = "1mo",
    feature_configs: list[dict] = None,
    hidden_layers: tuple[int, ...] = (64, 32),
    n_epochs: int = 500,
    lr: float = 1e-3,
    optim: str = "adam",
    device: str = get_device(),
    save_plot: bool = True,
    save_model: bool = True,
) -> tuple[nn.Module, dict]:

    # Default features - TODO: add dynamic feature code (config file input, not hardcoded)
    if feature_configs is None:
        feature_configs = [
            {'type': 'momentum', 'window': 1},
            {'type': 'momentum', 'window': 12},
            {'type': 'volatility', 'window': 3},
            {'type': 'volatility', 'window': 12},
        ]

    # Prepare data: data loaders, timeseries tensors of features (X_test_t) and returns (Y_test_t)
    train_loader, val_loader, X_test_t, Y_test_t, n_assets = prepare_data(
        data_path=data_path,
        tickers=tickers,
        data_period=data_period,
        interval=interval,
        feature_configs=feature_configs,
        device=device,
    )
    
    # Get input dimension from first batch
    X_sample, _ = next(iter(train_loader)) # EXPLAIN
    n_features = X_sample.shape[1]
    
    # Create model and trainer
    model = MLPModel(n_features, n_assets, *hidden_layers)
    trainer = Trainer(model, device=device)
    
    # Train
    model, output = trainer.train(
        train_loader, 
        val_loader, 
        n_epochs=n_epochs, 
        lr=lr, 
        optim=optim
    )
    
    # Evaluate on test set
    model.eval()
    loss_fn = torch.nn.MSELoss()
    with torch.inference_mode():
        Y_hat = model(X_test_t)
        test_mse = loss_fn(Y_hat, Y_test_t).item()
    
    print(f"Test MSE: {test_mse:.6f}")
    
    # Save artifacts
    now = datetime.now().strftime("%F_%H:%M:%S")
    train_out_dir = os.path.join(SAVE_DIR, now)

    try:
        os.makedirs(train_out_dir, exist_ok=False)
    except OSError:
        print(f"Directory(s) already exists in {train_out_dir}")
        breakpoint()
    
    if save_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(output["epochs"], output["avg_val_losses"], label="val_loss")
        plt.plot(output["epochs"], output["avg_train_losses"], label="train_loss")
        plt.title("Train & Val Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join(train_out_dir, f"loss-curve.png"))
        plt.close()
    
    if save_model:
        torch.save(model.state_dict(), os.path.join(train_out_dir, f"weights.pt"))
    
    metrics = {
        **output,
        "test_mse": test_mse,
    }

    hyperparams = {
        "batch_size": 32,
        "n_epochs": n_epochs,
        "lr": lr,
        "optim": optim,
        "hidden_layers": hidden_layers,
        "device": device,
    }

    Log(
        now,
        n_features, 
        data_period, 
        interval,
        hyperparams,
        test_mse,
        feature_configs
    ).save_log(train_out_dir)
    # save in models, models/{model_(date)}*/{model_(date).pt, model_(date).yaml, plot.png}*
    
    return model, metrics


def run_trainer_flexible(
    data_path: str = None,
    tickers: list[str] = None,
    data_period: str = "5y",
    interval: str = "1mo",
    feature_configs: list[dict] = None,
    hidden_layers: tuple[int, ...] = (32, 16),
    n_epochs: int = 500,
    lr: float = 1e-3,
    optim: str = "adam",
    device: str = get_device(),
    save_plot: bool = True,
    save_model: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Train a flexible model that can predict returns for ANY stock.
    
    This uses stacked data format where each sample is one asset at one time,
    allowing the model to learn patterns that generalize across assets.
    
    Args:
        Same as run_trainer(), but uses FlexibleMLPModel instead of MLPModel
    
    Returns:
        (model, metrics) where model can predict any stock
    """
    from .model import FlexibleMLPModel
    
    # Default features
    if feature_configs is None:
        feature_configs = [
            {'type': 'momentum', 'window': 1},
            {'type': 'momentum', 'window': 12},
            {'type': 'volatility', 'window': 3},
            {'type': 'volatility', 'window': 12},
        ]
    
    # Prepare data (same as before)
    extractor = DataExtractor()
    if data_path and not tickers:
        extractor.extract_csv(data_path)
    elif tickers:
        extractor.tickers = tickers
    else:
        raise ValueError("Must provide either data_path or tickers")
    
    timeseries = extractor.extract_yfinance(period=data_period, interval=interval)
    print(f"Loaded {len(extractor.tickers)} tickers")
    
    # Process data
    processor = DataProcessor(timeseries, date_col="Date", null_threshold=0.1)
    max_window = max([item["window"] for item in feature_configs])
    processor.clean_data().compute_returns().add_features(feature_configs).finalize(max_window)
    
    asset_cols = processor.get_asset_columns()
    print(f"Kept {len(asset_cols)} assets after filtering")
    
    # KEY DIFFERENCE: Use stacked format
    X, Y, metadata = processor.get_features_and_returns_stacked()
    n_features_per_asset = metadata['n_features_per_asset']
    
    print(f"\nStacked data shape:")
    print(f"  X: {X.shape} - each row is one asset at one time")
    print(f"  Y: {Y.shape} - predicted return for each")
    print(f"  Features per asset: {n_features_per_asset}")
    print(f"  Total samples: {X.shape[0]} = {metadata['n_times']} times × {metadata['n_assets']} assets")
    
    # Split data (same time-based split)
    X_train, X_val, X_test = split_train_data(X, 0.7)
    Y_train, Y_val, Y_test = split_train_data(Y, 0.7)
    
    print(f"\nTrain: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test:  X={X_test.shape}, Y={Y_test.shape}")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)
    
    # Create dataloaders (larger batch size since we have more samples)
    train_loader = DataLoader(CostDataset(X_train_t, Y_train_t), batch_size=256, shuffle=True)
    val_loader = DataLoader(CostDataset(X_val_t, Y_val_t), batch_size=256, shuffle=False)
    
    # KEY DIFFERENCE: Use FlexibleMLPModel with n_features_per_asset input
    model = FlexibleMLPModel(n_features_per_asset, *hidden_layers)
    print(f"\nModel architecture: {n_features_per_asset} inputs → {hidden_layers} → 1 output")
    
    # Train (same as before)
    trainer = Trainer(model, device=device)
    model, output = trainer.train(
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        lr=lr,
        optim=optim
    )
    
    # Evaluate on test set
    model.eval()
    loss_fn = torch.nn.MSELoss()
    with torch.inference_mode():
        Y_hat = model(X_test_t)
        test_mse = loss_fn(Y_hat, Y_test_t).item()
    
    print(f"\nTest MSE: {test_mse:.6f}")
    
    # Save artifacts
    now = datetime.now().strftime("%F_%H:%M:%S")
    train_out_dir = os.path.join(SAVE_DIR, f"flexible_{now}")
    
    try:
        os.makedirs(train_out_dir, exist_ok=False)
    except OSError:
        print(f"Directory already exists: {train_out_dir}")
    
    if save_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(output["epochs"], output["avg_val_losses"], label="val_loss")
        plt.plot(output["epochs"], output["avg_train_losses"], label="train_loss")
        plt.title("Train & Val Loss (Flexible Model)")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join(train_out_dir, "loss-curve.png"))
        plt.close()
    
    if save_model:
        torch.save(model.state_dict(), os.path.join(train_out_dir, "weights.pt"))
    
    metrics = {
        **output,
        "test_mse": test_mse,
    }
    
    hyperparams = {
        "batch_size": 256,
        "n_epochs": n_epochs,
        "lr": lr,
        "optim": optim,
        "hidden_layers": hidden_layers,
        "device": device,
        "model_type": "flexible",  # Mark as flexible model
    }
    
    # Save log
    Log(
        now,
        n_features_per_asset,  # Not total features, but per-asset
        data_period,
        interval,
        hyperparams,
        test_mse,
        feature_configs
    ).save_log(train_out_dir)
    
    print(f"\nModel saved to: {train_out_dir}")
    print(f"This model can now predict returns for ANY stock with {n_features_per_asset} features!")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = run_trainer_flexible(data_path="sp500-stocks.csv", n_epochs=500, lr=0.01, data_period="10y")
    breakpoint()
    print(metrics)


