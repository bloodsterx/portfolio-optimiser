import torch

class SPOPlusLoss:
    def __init__(self, oracle):
        self.oracle = oracle

    def __call__(self, c_hat: torch.tensor, c_true: torch.tensor, cov: torch.tensor, rf: float = 0.0):
        """
        SPO+ loss function for cost-based optimization.
        
        From Elmachtoub & Grigas (2022):
        Loss(c_hat) = max_{w in S} [c_true^T w - 2*c_hat^T w] + 2*c_hat^T w*(c_true) - c_true^T w*(c_true)
        
        where w*(c) is the optimal solution under cost vector c.
        
        Args:
            c_hat: Predicted costs (B x N), where costs = -expected_returns
            c_true: True costs (B x N), where costs = -expected_returns
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

        