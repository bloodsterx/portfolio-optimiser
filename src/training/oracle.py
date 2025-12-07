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
        cov: torch.Tensor,
        risk_av: float = 1.0,
        rf: float = 0.0,
        long_only: bool = True, # to be changed for more constraints
        device: str = "cpu",
        solver: str = "ECOS" # TODO
        # constraints
    ):
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda not available")
            device = "cpu"
        
        self.device = torch.device(device)
        self.cov = cov.to(device) # caller must pass in LW-shrunk covariance
        self.risk_av = risk_av
        self.rf = rf
        self.long_only = long_only

        self.cov_np = cov.cpu().numpy()
        self.n_assets = self.cov_np.shape[0]

    def __call__(self, C: torch.Tensor) -> torch.Tensor:
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


        """
        C_np = C.detach().cpu().numpy() # detach since cvxpy doesn't work with tensors
        batch_size = C_np.shape[0]

        W_batch = np.zeros_like(C_np)
        # TODO: super inefficient, can we do batch optimisation or use threads?
        for i in range(batch_size):
            mu = C_np[i] # N x 1 (expected return vector)
            w = cp.Variable(self.n_assets) 

            p_var = cp.quad_form(w, self.cov_np)
            p_ret = w@mu
            objective = cp.Minimize((0.5*self.risk_av)*p_var - p_ret)
            constraints = [
                w.sum() == 1,
                # TODO: add more based on user input
            ]

            if self.long_only:
                constraints.append(w >= 0)

            problem = cp.Problem(objective, constraints)

            try:
                problem.solve(solver=cp.ECOS) # if accuracy shite, use MOSEK? If MOSEK too slow, do batch optim

                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    W_batch[i] = w.value

                else:
                    print(f"Optimisation failed for sample {i} with status {problem.status}")
                    W_batch[i] = np.ones(self.n_assets) / self.n_assets # equal weight fallback

            except Exception as e:
                print(f"Error solving optimisation failed for sample {i}: {e}")
                W_batch[i] = np.ones(self.n_assets) / self.n_assets # equal weight fallback
        
        W_torch = torch.tensor(W_batch, dtype=C.dtype, device=self.device)
        return W_torch