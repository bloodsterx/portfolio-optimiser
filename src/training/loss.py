import torch

class SPOPlusLoss:
    def __init__(self, oracle):
        self.oracle = oracle

    def __call__(self, c_hat: torch.tensor, c_true: torch.tensor, cov: torch.tensor, rf: float = 0.0):
        """
        SPO+ loss function.
        
        xi_s(c) := max_{w in S} [cT w - 2c_hatT w] + 2c_hatT w*(c) - w_true
        
        Args:
            c_hat: Predicted expected returns (B x N)
            c_true: True expected returns (B x N)
            cov: Covariance matrix (N x N)
            rf: Risk-free rate (scalar or tensor)
        """
        with torch.no_grad():
            w_true = self.oracle(c_true, cov, rf)

        c_tilted = c_true - 2 * c_hat
        w_tilted = self.oracle(c_tilted, cov, rf)

        T1 = (c_tilted * w_tilted).sum(dim=1)
        T2 = 2 * (c_hat * w_true).sum(dim=1)
        T3 = (c_true * w_true).sum(dim=1)

        loss = (T1 + T2 - T3).mean()
        return loss

        