import numpy as np

def MVO(ret: np.ndarray, cov: np.ndarray, A: float, rf: float, constraints: list[str] = [""]):
    """
    Perform Mean-Variance Optimization (MVO) with a risk-free asset.

    Computes the optimal portfolio weights for risky assets that maximize expected utility:
        U(w) = w.T @ μ - (A/2) * w.T @ Σ @ w
    
    With a risk-free asset available, the closed-form solution is:
        w* = (1/A) * Σ^(-1) @ (μ - rf·1)

    Args:
        ret (np.ndarray): Expected returns vector for risky assets, shape (n_assets,)
        cov (np.ndarray): Covariance matrix of risky asset returns, shape (n_assets, n_assets)
        A (float): Risk aversion coefficient (A > 0). Higher values indicate greater risk aversion.
        rf (float): Risk-free rate of return.
        constraints (list[str]): List of constraint strings (not yet implemented).

    Returns:
        np.ndarray: Optimal weights for risky assets, shape (n_assets,). 
                    Weights may not sum to 1; remainder should be invested in risk-free asset.
                    If sum(w*) > 1, this implies borrowing at the risk-free rate (leverage).

    Notes:
        - Current implementation assumes no constraints (e.g., allows short selling).
        - Constraints parameter is a placeholder for future implementation.
        - The formula assumes the covariance matrix is invertible (positive definite).
    """
    
    # ignore constraints for now, later implement a constraint parser
    # which reads in constraints as strings... Or maybe a dictionary... figure out a DS

    return np.pow(A, -1) * np.linalg.inv(cov)@(ret - np.ones_like(ret) * rf)




    
