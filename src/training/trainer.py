import torch.nn as nn
import torch
import numpy as np
import polars as pl
from torch.utils.data import DataLoader

from .model import MLPModel
from ..data.data import DataExtractor, CostDataset
from ..data.covariance import compute_rolling_covariance
from ..data.features import Features
from .loss import SPOPlusLoss
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
        oracle_typ="MVO",
        optim="adam", 
        n_epochs=100, 
        lr = 1e-3, 
        loss_type="MSE",
        risk_av=5.0,
    ):

        # Initialize optimizer
        match optim:
            case "adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            case "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
        # Initialize oracle once (for SPO+)
        oracle = None
        match oracle_typ:
            case "MVO":
                oracle = MVO(
                    risk_av=risk_av,
                    long_only=True,
                    device=self.device
                )

        # Initialize loss function once
        loss_fn = None
        match loss_type:
            case "MSE":
                loss_fn = nn.MSELoss()
            case "SPO+":
                loss_fn = SPOPlusLoss(oracle=oracle)

        output = []

        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.0
            n_train_samples = 0

            for X_batch, C_batch, Sigma, rf in train_dataloader:
                # Move all data to device
                X_batch = X_batch.to(self.device)
                C_batch = C_batch.to(self.device)
                Sigma = Sigma.to(self.device)

                # if rf is variable or constant (e.g. using historical US 10 year T-bill )
                rf = rf.to(self.device) if isinstance(rf, torch.Tensor) else rf

                # 1. forward pass
                c_hat = self.model(X_batch)

                # 2. loss calc (pass Sigma and rf for SPO+)
                match loss_type:
                    case "SPO+":
                        loss = loss_fn(c_hat, C_batch, Sigma, rf)
                    case _:
                        loss = loss_fn(c_hat, C_batch)

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
            val_mse = 0.0
            val_regret = 0.0
            n_val_samples = 0

            # validation 
            with torch.inference_mode():
                for X_batch, C_batch, Sigma, rf in val_dataloader:
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)
                    Sigma = Sigma.to(self.device)
                    rf = rf.to(self.device) if isinstance(rf, torch.Tensor) else rf

                    c_hat = self.model(X_batch)

                    match loss_type:
                        case "SPO+":
                            loss = loss_fn(c_hat, C_batch, Sigma, rf)
                        case _:
                            loss = loss_fn(c_hat, C_batch)

                    val_loss += loss.item() * X_batch.size(0)

                    # calc mse-loss
                    mse = ((c_hat - C_batch) ** 2).mean()
                    val_mse += mse.item() * X_batch.size(0)

                    # calc 'regret' (decision quality - cost of choosing port A over B)
                    w_hat = oracle(c_hat, Sigma, rf)
                    w_true = oracle(C_batch, Sigma, rf)

<<<<<<< HEAD
                    expr_hat = torch.einsum('Bi, Bij, Bj -> B', w_hat, Sigma, w_hat)
                    expr_true = torch.einsum('Bi, Bij, Bj -> B', w_true, Sigma, w_true)

                    util_hat = (C_batch * w_hat).sum(dim=1) - 0.5 * risk_av * (expr_hat)
                    util_true = (C_batch * w_true).sum(dim=1) - 0.5 * risk_av * (expr_true)
=======
                    # Utility = -cost - risk_penalty (since C_batch is costs, -C_batch gives returns)
                    util_hat = (-C_batch * w_hat).sum(dim=1) - 0.5 * risk_av * (w_hat @ Sigma * w_hat).sum(dim=1)
                    util_true = (-C_batch * w_true).sum(dim=1) - 0.5 * risk_av * (w_true @ Sigma * w_true).sum(dim=1)
>>>>>>> 0386ba3 (refactor: cost is actually negative of returns now, changed applied for staying consistent with the literature)
                    
                    regret = (util_true - util_hat).mean()
                    val_regret += regret.item() * X_batch.size(0) #

                    n_val_samples += X_batch.size(0)

            avg_val_loss = val_loss / n_val_samples
            avg_val_mse = val_mse / n_val_samples
            avg_val_regret = val_regret / n_val_samples

            if epoch % 5 == 0:
                output.append({
                    "epoch": epoch,
                    "train loss": avg_train_loss,
                    "val Loss": avg_val_loss,
                    "val MSE": avg_val_mse,
                    "val regret": avg_val_regret,
                })

                print(f"Epoch {epoch:3d} | "
                    f"train loss: {avg_train_loss:.4f} | "
                    f"val loss: {avg_val_loss:.4f} | "
                    f"val MSE: {avg_val_mse:.4f} | "
                    f"val regret: {avg_val_regret:.4f}")


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


def run():
    # step 1. extract the features from the data
    import csv
    
    with open("sp500-stocks.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        tickers = [row[0].strip() for row in reader if row][:10]
    
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
    
    # prices -> returns conversion
    asset_cols = [col for col in timeseries_pl.columns if col != "Date"]
    returns_pl = timeseries_pl.with_columns([
        pl.col(col).pct_change().alias(col) for col in asset_cols
    ])

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
    # Start with returns_pl (has Date + all asset returns)
    combined = returns_pl.clone()

    # Join each feature DataFrame on Date
    for feature_df in [mom1m, mom12m, volatility1m, volatility12m]:
        combined = combined.join(feature_df, on="Date", how="left")
    
    combined = combined.drop_nulls()
    
    print(f"Combined features shape: {combined.shape}")
    print(f"Columns: {combined.columns[:10]}...")
    
    feature_cols = [col for col in combined.columns 
                    if col not in ["Date"] + asset_cols]

    X = combined.select(feature_cols).to_numpy()
    # Convert returns to costs (c = -returns) to match SPO+ paper formulation
    returns = combined.select(asset_cols).to_numpy()
    C = -returns  # Cost formulation: minimize costs = maximize returns
    
    print(f"X shape: {X.shape}, C shape: {C.shape}")

    # step 2. prepare data into splits, into datasets and create loader objects 
    X_train, X_val, X_test = Trainer.split_train_data(X, 0.7)
    C_train, C_val, C_test = Trainer.split_train_data(C, 0.7)
    cov_train = compute_rolling_covariance(C_train, window=3, method='ledoit_wolf')
    cov_val = compute_rolling_covariance(C_val, window=3, method='ledoit_wolf')
    cov_test = compute_rolling_covariance(C_test, window=3, method='ledoit_wolf')
    
    print(f"cov_train shape: {cov_train.shape}, cov_val shape: {cov_val.shape}, cov_test shape: {cov_test.shape}")
    print(f"cov_train head: {cov_train[:5]}, cov_val head: {cov_val[:5]}, cov_test head: {cov_test[:5]}")
    
    print(f"Train: X={X_train.shape}, C={C_train.shape}")
    print(f"Val: X={X_val.shape}, C={C_val.shape}")
    print(f"Test: X={X_test.shape}, C={C_test.shape}")
    
    # Calculate covariance matrix from training data
    rf = 0.02
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    C_train_t = torch.tensor(C_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    C_val_t = torch.tensor(C_val, dtype=torch.float32)
    cov_train_t = torch.tensor(cov_train, dtype=torch.float32)
    cov_val_t = torch.tensor(cov_val, dtype=torch.float32)

    # setup dataloader objects
    train_dataset = CostDataset(X_train_t, C_train_t, cov_train_t, rf)
    val_dataset = CostDataset(X_val_t, C_val_t, cov_val_t, rf)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Created dataloaders with {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # step 3. train!
    model = MLPModel(X_train_t.shape[1], len(asset_cols))
    trainer = Trainer(model)
    model, output = trainer.train(train_loader, val_loader, loss_type="SPO+")

    print(output)
    
if __name__ == "__main__":
    run()
    


