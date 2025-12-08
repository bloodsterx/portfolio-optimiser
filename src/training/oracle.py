import torch
import numpy as np
import cvxpy as cp

class MVO:
    """
    Markowitz oracle

    Given cost vector c (shape B x N), uses
        mu = -c
        w_raw ∝ Σ^{-1} mu (w = A^{-1}*Σ^{-1}*mu => w propto invSigma @ mu)
    and normalises rows to sum to 1

    Optional constraint (to be changed later for more flexibility): long only

    Covariance matrix should be regularised with LW or with a small ridge before passing into oracle
    Future:
    - allow for time-varying Cov. GARCH? MixtureOfExperts forests which classify regimes and provide different covariance estiamtes for each regime

    """
    def __init__(
        self,
        risk_av: float = 1.0,
        long_only: bool = True,
        device: str = "cpu",
        solver: str = "ECOS" # TODO
        # constraints
    ):
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda not available")
            device = "cpu"
        
        self.device = torch.device(device)
        self.risk_av = risk_av
        self.long_only = long_only
        self.solver = solver

    def __call__(self, C: torch.Tensor, cov: torch.Tensor, rf: float = 0.0) -> torch.Tensor:
        """
         for i in range batch

        get the i'th expected return (-C_np[i])

        Define the optimisation variable (optim over weights 'w')

        setup the optimisation problem:
        - variance = cp.quad_form
        - portfolio return = mu @ w
        - objective = minimise (-(mu@w - 0.5Aw@Sigma@wT))
        - constraints = [cp.sum(variable) = 1,w >= 0]

        define the problem cp.Problem()
        Solve portfolio optimization for a batch of expected returns.
        
        Args:
            C: Expected returns tensor (B x N)
            cov: Covariance matrix (N x N), supports time-varying covariance
            rf: Risk-free rate (scalar or tensor), supports time-varying rf
        
        Returns:
            W: Optimal portfolio weights (B x N)
        """
        # Move cov to device and convert to numpy
        C_np = C.detach().cpu().numpy() # detach since cvxpy doesn't work with tensors
        cov_np = cov.detach().cpu().numpy()
        n_assets = cov.shape[0]
        
        batch_size = C_np.shape[0]

        W_batch = np.zeros_like(C_np)
        # TODO: super inefficient, can we do batch optimisation or use threads?
        for i in range(batch_size):
            mu = C_np[i]  # N x 1 (expected return vector)
            w = cp.Variable(n_assets) 

            p_var = cp.quad_form(w, cov_np)
            p_ret = w @ mu
            objective = cp.Minimize((0.5 * self.risk_av) * p_var - p_ret)
            constraints = [
                w.sum() == 1,
                # TODO: add more based on user input
            ]

            if self.long_only:
                constraints.append(w >= 0)

            problem = cp.Problem(objective, constraints)

            try:
                problem.solve(solver=self.solver)

                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    W_batch[i] = w.value

                else:
                    print(f"Optimisation failed for sample {i} with status {problem.status}. Fallback to equal weights")
                    # or should I just abort
                    W_batch[i] = np.ones(n_assets) / n_assets  # equal weight fallback

            except Exception as e:
                print(f"Error solving optimisation for sample {i}: {e}")
                W_batch[i] = np.ones(n_assets) / n_assets  # equal weight fallback
        
        W_torch = torch.tensor(W_batch, dtype=C.dtype, device=self.device)
        return W_torch