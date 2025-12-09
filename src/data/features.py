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
                        .rolling_sum(window_size=window)  # sum over window periods
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
                    .rolling_std(window_size=window)  # rolling standard deviation of returns
                    .shift(1)  # lag by 1 period
                ).alias(f"{col}_realvol{window}{units.lower()}")
                for col in self.asset_cols
            ]
        )

        return result

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
                            (pl.col(col) * pl.col(bench_col)).rolling_mean(window_size=window)
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
