import numpy as np


def compute_rolling_covariance(
    returns: np.ndarray,
    window: int,
    method: str = 'ledoit_wolf'
) -> np.ndarray:
    """
    Compute rolling covariance matrices from returns data.

    For each time t, computes covariance from returns in [t-window+1, t].
    Eliminates look-ahead bias - cov at time t only uses data up to t.

    Args:
        returns: Array of shape (T, n_assets) containing asset returns
        window: Number of periods to use for covariance estimation
        method: Estimation method:
            - 'sample': Standard sample covariance with small regularization
            - 'ledoit_wolf': Ledoit-Wolf shrinkage (recommended for finance)

    Returns:
        Array of shape (T, n_assets, n_assets) with covariance matrices.
        Early periods (t < window-1) contain zeros (warm-up phase).
    """
    if method == 'ledoit_wolf':
        from sklearn.covariance import LedoitWolf

    T, n_assets = returns.shape

    # 3 dimensional T x (n x n)
    cov_matrices = np.zeros((T, n_assets, n_assets))

    if method == 'ledoit_wolf':
        lw = LedoitWolf()

    for t in range(T):
        if t < window - 1:
            # warm up phase - not enough data to calculate 'window'-sized rolled covariance
            continue
        
        # rolling window [t-window+1, t] inclusive
        start_idx = t - window + 1
        window_returns = returns[start_idx:t+1]

        if method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage 
            cov_matrices[t] = lw.fit(window_returns).covariance_
        else:
            # add ridge to regularise matrix
            cov_matrices[t] = np.cov(window_returns, rowvar=False)
            cov_matrices[t] += np.eye(n_assets) * 1e-6

    return cov_matrices

