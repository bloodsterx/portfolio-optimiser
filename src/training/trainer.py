import torch.nn as nn
import torch
from .loss import SPOPlusLoss
from .oracle import MVO

def train(
    model, 
    dataloader,
    cov,
    oracle_typ="MVO",
    optim="adam", 
    n_epochs=100, 
    lr = 1e-3, 
    device="cpu", 
    loss="MSE",
    risk_av=5,
    rf=0.0
):

    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available on this machine"

    model.to(device)
    optimizer = None    
    loss_fn = None
    oracle = None

    match optim:
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    match oracle_typ:
        case "MVO":
            oracle = MVO

    match loss:
        case "MSE":
            loss_fn = nn.MSELoss()
        case "SPO+":
            loss_fn = SPOPlusLoss(oracle=oracle, cov=cov, risk_av=risk_av, rf=rf)

    output = []

    for epoch in range(n_epochs):
        # in train mode

        model.train()
        cum_loss = 0.0

        for X_batch, C_batch, Sigma, rf in dataloader:
            # move data to device? (what does this do)
            X_batch = X_batch.to(device)
            C_batch = C_batch.to(device)

            # 1. forward pass
            c_hat = model(X_batch)

            # 2. Calculate loss
            # For SPO+, the oracle is called inside the loss function
            # For MSE, just compute prediction error
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
    
    return model, output


   
        


