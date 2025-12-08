import torch.nn as nn
import torch
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
                rf = rf.to(self.device) if isinstance(rf, torch.Tensor) else rf

                # 1. forward pass
                c_hat = self.model(X_batch)

                # 2. Calculate loss (pass Sigma and rf for SPO+)
                if loss_type == "SPO+":
                    loss = loss_fn(c_hat, C_batch, Sigma, rf)
                else:
                    loss = loss_fn(c_hat, C_batch)

                # 4. clear old gradients
                optimizer.zero_grad()  

                # 4. Backpropagation: compute gradients
                loss.backward()  # Fixed typo: backwards -> backward
                
                # 5. Update model parameters
                optimizer.step()

                cum_loss += loss.item() * X_batch.size(0)

                output.append(
                    f"Epoch [{epoch}]: cumulative loss={cum_loss}"
                )
        
        return self.model, output



