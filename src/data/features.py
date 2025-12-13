import polars as pl
import numpy as np

"""
Thoughts

Key indicators (GKX, 2020)
- Momentum
- Liquidity
- Volatility

Iteration 1 indicators
- mom1m (end of previous month return indicator of short horizon reversal)
- mom12m 
- 3m realised vol
- 12m realised vol
- rolling market beta

no liquidity measures... How to do that for ff49 without microstructure data?

Stuff to test
- lagged IV instead of realised. Thought process: investors do not make decisions based off of realized volatility (unobservable). 
IV is observable from option market flows and dynamics. Investors actually use IV as a forward looking measure of vol, so it should be more relevant
as a feature
- GARCH

"""


class Features:
    def __init__(self, data: pl.DataFrame, date_col: str = "Date"):
        """
        Expect a T x N matrix (periods, returns)

        Args:
            data: DataFrame with columns [Date, Asset1, Asset2, ..., AssetN]
                  where values are returns (in percentage or decimal)
            date_col: Name of the date column
        """
        self.data = data
        self.date_col = date_col
        self.asset_cols = [col for col in data.columns if col != date_col]
        self.n_assets = len(self.asset_cols)

    def mom(self, window, units="m") -> pl.DataFrame:
        """
        Returns a matrix of lagged compound returns (momentum)

        Args:
            window: int - interval over which to calculate momentum. Units in terms of input data.
            E.g. window=1 => 1-period lagged return, window=12 => compounded return over 12 periods, lagged by 1
            units: (d, w, m, y)

        Returns:
            DataFrame with Date and momentum columns for each asset, lagged by 1 period
        """
        match units.lower():
            case "d" | "w" | "m" | "y":
                pass
            case _:
                print(f"{'='*10} Exception Here {'='*10}")
                print("Units must be either d, w, m or y")
                return

        result = self.data.select(
            pl.col(self.date_col),
            *[
                (
                    # Compound return = exp(sum(log(1+r))) - 1, applied to rolling window, then lagged
                    (
                        pl.col(col).log1p()  # log(1 + return)
                        # sum over window periods
                        .rolling_sum(window_size=window)
                        .exp() - 1.0  # exponentiate and subtract 1 to get compound return
                    ).shift(1)  # lag by 1 period
                ).alias(f"{col}_mom{window}{units.lower()}")
                for col in self.asset_cols
            ]
        )
        return result

    def volatility(self, window, units="m") -> pl.DataFrame:
        """
        Calculate rolling realized volatility (standard deviation of returns)

        Args:
            window: int - rolling window size for volatility calculation
            units: (d, w, m, y)

        Returns:
            DataFrame with Date and volatility columns for each asset, lagged by 1 period
        """
        match units.lower():
            case "d" | "w" | "m" | "y":
                pass
            case _:
                print(f"{'='*10} Exception Here {'='*10}")
                print("Units must be either d, w, m or y")
                return

        result = self.data.select(
            pl.col(self.date_col),
            *[
                (
                    pl.col(col)
                    # rolling standard deviation of returns
                    .rolling_std(window_size=window)
                    .shift(1)  # lag by 1 period
                ).alias(f"{col}_realvol{window}{units.lower()}")
                for col in self.asset_cols
            ]
        )

        return result

    def compute_rolling_covariance(
        returns: np.ndarray,
        window: int,
        min_periods: int = None,
        method: str = 'ledoit_wolf'
    ) -> np.ndarray:
        """
        Compute rolling covariance matrices from returns data.

        For each time t, computes covariance from returns in [t-window+1, t].
        eliminates look-ahead bias - cov at time t only uses data up to t.

        Args:
            returns: Array of shape (T, n_assets) containing asset returns
            window: Number of periods to use for covariance estimation
            min_periods: Minimum periods required (defaults to window)
            method: Estimation method:
                - 'sample': Standard sample covariance with small regularization
                - 'ledoit_wolf': Ledoit-Wolf shrinkage (recommended for finance)

        Returns:
            Array of shape (T, n_assets, n_assets) with covariance matrices.
            Early periods with insufficient data use identity matrix scaled by avg variance.
        """
        if method == 'ledoit_wolf':
            from sklearn.covariance import LedoitWolf

        T, n_assets = returns.shape
        min_periods = min_periods or window

        cov_matrices = np.zeros((T, n_assets, n_assets))

        if method == 'ledoit_wolf':
            lw = LedoitWolf()

        for t in range(T):
            if t < min_periods - 1:
                # Not enough data - use scaled identity as fallback
                # Estimate variance from available data or use small default
                if t > 0:
                    available = returns[:t+1]
                    avg_var = np.var(available, axis=0).mean()
                else:
                    avg_var = 0.01  # Default variance
                cov_matrices[t] = np.eye(n_assets) * avg_var
            else:
                # Use rolling window [t-window+1, t] inclusive
                start_idx = max(0, t - window + 1)
                window_returns = returns[start_idx:t+1]

                if method == 'ledoit_wolf':
                    # Ledoit-Wolf shrinkage (already regularized, no need for extra epsilon)
                    cov_matrices[t] = lw.fit(window_returns).covariance_
                else:
                    # Sample covariance with small regularization
                    cov_matrices[t] = np.cov(window_returns, rowvar=False)
                    cov_matrices[t] += np.eye(n_assets) * 1e-6

        return cov_matrices

    def beta(self, window, bench: pl.DataFrame, bench_col: str = "benchmark", units="m") -> pl.DataFrame:
        """Calculates the rolling beta of asset to benchmark. Defaults to market portfolio proxy (FF49)

        Args:
            window (int): interval over which to calculate beta. Units in terms of input data.
            bench (pl.DataFrame): benchmark to calculate beta against. Should have columns [Date, benchmark_col]
            bench_col (str): name of the benchmark return column. Defaults to "benchmark".
            units (str, optional): (d, w, m, y). Defaults to "m".

        Returns:
            DataFrame with Date and beta columns for each asset, lagged by 1 period
        """
        match units.lower():
            case "d" | "w" | "m" | "y":
                pass
            case _:
                print(f"{'='*10} Exception Here {'='*10}")
                print("Units must be either d, w, m or y")
                return

        # join bench and data. ASSUMES NO CODING ERRORS. TODO: clean data before calculating
        data_with_bench = self.data.join(bench.select(pl.col(self.date_col), pl.col(bench_col)),
                                         on=self.date_col,
                                         how="left")

        # Calculate rolling beta for each asset: beta = cov(asset, bench) / var(bench)
        result = data_with_bench.select(
            pl.col(self.date_col),
            *[
                (
                    (
                        # Rolling covariance: E[XY] - E[X]E[Y]
                        (
                            (pl.col(col) * pl.col(bench_col)
                             ).rolling_mean(window_size=window)
                            -
                            (pl.col(col).rolling_mean(window_size=window) *
                             pl.col(bench_col).rolling_mean(window_size=window))
                        )
                        /
                        # Divide by rolling variance of benchmark
                        pl.col(bench_col).rolling_var(window_size=window)
                    )
                    # Lag the entire beta by 1 period
                    .shift(1)
                ).alias(f"{col}_beta{window}{units.lower()}")
                for col in self.asset_cols
            ]
        )

        return result
