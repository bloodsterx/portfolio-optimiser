from model import MLPModel as model
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

   
        


