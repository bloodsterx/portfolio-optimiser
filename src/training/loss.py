import torch

class SPOPlusLoss:
    def __init__(self, oracle):
        self.oracle: oracle = oracle

    def __call__(self, c_hat: torch.tensor, c_true: torch.tensor):
        """
        xi_s(c) := max_{w in S} [cT w - 2c_hatT w] + 2c_hatT w*(c) - w_true
        """
        with torch.no_grad(): # expect tensor inputs with gradient tracking (we set)
            w_true = self.oracle(c_true)

        c_tilted = c_true - 2 * c_hat
        w_tilted = self.oracle(c_tilted)

        T1 = (c_tilted * w_tilted).sum(dim=1)
        T2 = 2 * (c_hat * w_true).sum(dim=1)
        T3 =  (c_true * w_true).sum(dim=1)

        loss = (T1 + T2 - T3).mean()
        return loss

        