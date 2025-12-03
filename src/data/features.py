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

no liquidity measures... How to do that for ff30 without microstructure data?
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
        Returns a matrix of lagged-1m stock returns

        Args:
            window: int - interval over which to calculate momentum. Units in terms of input data.
            E.g. window=12 => compounded return over 12 rows. If rows correspond to months, then window=12 => 12 month momentum

            units: (d, w, m, y)
        """
        match units.lower():
            case "d" | "w" | "m" | "y":
                pass
            case _:
                print(f"{'='*10} Exception Here {'='*10}")

        result = self.data.select(
            pl.col(self.date_col),
            *[
                (
                    pl.col(col).shift(1).log1p()
                    .rolling_sum(window_size=window)
                    .exp() - 1.0
                )
                .alias(f"{col}_mom{window}{units}") # just need to shift 1 month to apply lag
                for col in self.asset_cols
            ]
        )
        return result
        
    def mom1m(self) -> pl.DataFrame:
        """
        Returns a matrix of lagged-1m stock returns
        """
        result = self.data.select(
            pl.col(self.date_col),
            *[
                (
                    pl.col(col).shift(1).log1p()
                    .rolling_sum(window_size=)
                )
                .alias(f"{col}_mom1m") # just need to shift 1 month to apply lag
                for col in self.asset_cols
            ]
        )
        return result
    
    def mom12m(self) -> pl.DataFrame:
        """
        Returns a matrix of 1m-lagged, 12m momentum (12-1 momentum strategy)
        """
        result = self.data.select(
            self.date_col,
            *[
                (
                    pl.col(col).shift(1).log1p() # sum logs then exponentiate to avoid under/overflow
                    .rolling_sum(window_size=11) # includes current row and previous 11 rows = 12 rows
                    .exp() - 1.0
                ).alias(f"{col}_mom12m")
                for col in self.asset_cols 
            ]
        )
        return result

    def vol3m(self)
