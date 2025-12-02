import torch.nn as nn
import torch

def train(
    model, 
    dataloader, 
    n_epochs=100, 
    lr = 1e-3, 
    device="cpu", 
    optim="adam", 
    loss="MSE"
):

    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available on this machine"

    model.to(device)
    optimizer = None    
    loss_fn = None

    match optim:
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    match loss:
        case "MSE":
            loss_fn = nn.MSELoss()
        case "SPO+":
            # TODO
            pass

    output = []

    for epoch in range(n_epochs):
        # in train mode

        model.train()
        cum_loss = 0.0

        for X_batch, C_batch in dataloader:
            # move data to device? (what does this do)
            X_batch = X_batch.to(device)
            C_batch = C_batch.to(device)

            # 1. forward pass
            c_hat = model(X_batch)

            # 2. calc (short for calculator btw (short for by the way)) loss
            loss = loss_fn(c_hat, C_batch)

            # 3. clear old gradients
            optimizer.zero_grad()  

            # 4. backprop
            loss.backwards() # traverse backprop graph and recomputes gradients for each node
            optimizer.step() # applies the gradient updates

            cum_loss += loss.item() * X_batch.size(0)

            output.append(
                f"Epoch [{epoch}]: cumulative loss={cum_loss}"
            )
    
    return model, output


   
        


