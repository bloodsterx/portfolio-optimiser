import torch.nn as nn
import torch
import numpy as np
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
        dataloader,
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
            cum_loss = 0.0

            for X_batch, C_batch, Sigma, rf in dataloader:
                # Move all data to device
                X_batch = X_batch.to(self.device)
                C_batch = C_batch.to(self.device)
                Sigma = Sigma.to(self.device)

                # if rf is variable or constant (e.g. using historical US 10 year T-bill )
                rf = rf.to(self.device) if isinstance(rf, torch.Tensor) else rf

                # 1. forward pass
                c_hat = self.model(X_batch)

                # 2. Calculate loss (pass Sigma and rf for SPO+)
                match loss_type:
                    case "SPO+":
                        loss = loss_fn(c_hat, C_batch, Sigma, rf)
                    case _:
                        loss = loss_fn(c_hat, C_batch)

                # 3. clear old gradients
                optimizer.zero_grad()  

                # 4. Backpropagation: compute gradients
                loss.backward()
                
                # 5. Update model parameters
                optimizer.step()

                cum_loss += loss.item() * X_batch.size(0)

            self.model.eval()
            test_mse = 0.0
            test_regret = 0.0
            n_test_samples = 0

            with torch.inference_mode():
                for X_batch, C_batch, Sigma, rf in dataloader:
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)
                    Sigma = Sigma.to(self.device)
                    rf = rf.to(self.device) if isinstance(rf, torch.Tensor) else rf

                    c_hat = self.model(X_batch)
                    
                    mse = ((c_hat - C_batch) ** 2).mean()
                    test_mse += mse.item() * X_batch.size(0)

                    w_hat = oracle(c_hat, Sigma, rf)
                    w_true = oracle(C_batch, Sigma, rf)

                    util_hat = (C_batch * w_hat).sum(dim=1) - 0.5 * risk_av * (w_hat @ Sigma * w_hat).sum(dim=1)
                    util_true = (C_batch * w_true).sum(dim=1) - 0.5 * risk_av * (w_true @ Sigma * w_true).sum(dim=1)
                    regret = (util_true - util_hat).mean()
                    test_regret += regret.item() * X_batch.size(0)

                    n_test_samples += X_batch.size(0)

            avg_train_loss = cum_loss / n_test_samples
            avg_test_mse = test_mse / n_test_samples
            avg_test_regret = test_regret / n_test_samples

            if epoch % 5 == 0:
                output.append({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "test_mse": avg_test_mse,
                    "test_regret": avg_test_regret,
                })

        return self.model, output

    def split_train_data(self, data: np.ndarray, split: float) -> tuple[np.ndarray]:
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

    


