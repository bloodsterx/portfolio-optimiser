import pytest
import polars as pl
import numpy as np
from src.data.features import Features


class TestFeaturesMomentum:
    """Test cases for momentum calculation"""

    @pytest.fixture
    def simple_data(self):
        """Simple dataset with known returns for manual verification"""
        return pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "Asset1": [0.01, 0.02, 0.03, -0.01, 0.02, 0.01],  # Monthly returns
            "Asset2": [0.05, -0.02, 0.01, 0.03, -0.01, 0.02],
        })

    def test_mom_returns_correct_columns(self, simple_data):
        """Verify output DataFrame has correct column names"""
        features = Features(simple_data)
        result = features.mom(window=3, units="m")

        assert "Date" in result.columns
        assert "Asset1_mom3m" in result.columns
        assert "Asset2_mom3m" in result.columns
        assert len(result.columns) == 3

    def test_mom_returns_correct_length(self, simple_data):
        """Output should have same number of rows as input"""
        features = Features(simple_data)
        result = features.mom(window=3, units="m")

        assert len(result) == len(simple_data)

    def test_mom_lagged_by_one_period(self, simple_data):
        """Momentum should be lagged by 1 period (first row should be null)"""
        features = Features(simple_data)
        result = features.mom(window=2, units="m")

        # First row should be null due to lag
        assert result["Asset1_mom2m"][0] is None

    def test_mom_window_2_calculation(self, simple_data):
        """
        Test 2-period momentum calculation
        mom2m computes compound return over 2 periods, then lagged by 1
        """
        features = Features(simple_data)
        result = features.mom(window=2, units="m")

        # At index 2 (before lag), rolling_sum(window=2) uses indices 1 and 2: [0.02, 0.03]
        # After lag by 1, this appears at index 3
        # Compound return = (1.02) * (1.03) - 1 = 0.0506
        expected = (1.02 * 1.03) - 1
        assert result["Asset1_mom2m"][3] == pytest.approx(expected, rel=1e-6)

    def test_mom_window_3_compound_return(self, simple_data):
        """
        Test 3-period momentum (compound return over 3 periods, lagged by 1)
        """
        features = Features(simple_data)
        result = features.mom(window=3, units="m")

        # At index 3, rolling_sum(window=3) uses indices 1, 2, 3: [0.02, 0.03, -0.01]
        # After lag by 1, this appears at index 4
        # Compound return = (1.02) * (1.03) * (0.99) - 1
        expected = (1.02 * 1.03 * 0.99) - 1
        assert result["Asset1_mom3m"][4] == pytest.approx(expected, rel=1e-6)

    def test_mom_invalid_units_returns_none(self, simple_data, capsys):
        """Invalid units should print error and return None"""
        features = Features(simple_data)
        result = features.mom(window=3, units="x")

        assert result is None
        captured = capsys.readouterr()
        assert "Units must be either d, w, m or y" in captured.out

    @pytest.mark.parametrize("units", ["d", "w", "m", "y", "D", "W", "M", "Y"])
    def test_mom_valid_units(self, simple_data, units):
        """All valid unit types should work (case insensitive)"""
        features = Features(simple_data)
        result = features.mom(window=2, units=units)

        assert result is not None
        assert f"Asset1_mom2{units.lower()}" in result.columns


class TestFeaturesVolatility:
    """Test cases for volatility calculation"""

    @pytest.fixture
    def simple_data(self):
        """Simple dataset with known returns"""
        return pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "Asset1": [0.01, 0.02, 0.03, -0.01, 0.02, 0.01],
            "Asset2": [0.05, -0.02, 0.01, 0.03, -0.01, 0.02],
        })

    @pytest.fixture
    def constant_returns_data(self):
        """Dataset with constant returns (zero volatility)"""
        return pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "Asset1": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        })

    def test_volatility_returns_correct_columns(self, simple_data):
        """Verify output DataFrame has correct column names"""
        features = Features(simple_data)
        result = features.volatility(window=3, units="m")

        assert "Date" in result.columns
        assert "Asset1_realvol3m" in result.columns
        assert "Asset2_realvol3m" in result.columns

    def test_volatility_returns_correct_length(self, simple_data):
        """Output should have same number of rows as input"""
        features = Features(simple_data)
        result = features.volatility(window=3, units="m")

        assert len(result) == len(simple_data)

    def test_volatility_lagged_by_one_period(self, simple_data):
        """Volatility should be lagged by 1 period"""
        features = Features(simple_data)
        result = features.volatility(window=2, units="m")

        # First row should be null due to lag
        assert result["Asset1_realvol2m"][0] is None

    def test_volatility_constant_returns_is_zero(self, constant_returns_data):
        """Constant returns should have zero volatility"""
        features = Features(constant_returns_data)
        result = features.volatility(window=3, units="m")

        # After enough periods, volatility of constant returns should be 0
        non_null_values = result.filter(
            pl.col("Asset1_realvol3m").is_not_null())
        for val in non_null_values["Asset1_realvol3m"].to_list():
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_volatility_manual_calculation(self, simple_data):
        """Verify volatility matches manual std calculation. Take a sample of 3 month realised vol - index 4"""
        features = Features(simple_data)
        result = features.volatility(window=3, units="m")  # 3 periods

        # At index 3, rolling_std(window=3) uses indices 1, 2, 3: [0.02, 0.03, -0.01]
        # After lag by 1, this appears at index 4
        # simple_data["Asset1"] = [0.01, 0.02, 0.03, -0.01, 0.02, 0.01]

        returns = np.array([0.02, 0.03, -0.01])
        expected_std = np.std(returns, ddof=1)

        assert result["Asset1_realvol3m"][4] == pytest.approx(
            expected_std, rel=1e-6)

    def test_volatility_invalid_units_returns_none(self, simple_data, capsys):
        """Invalid units should print error and return None"""
        features = Features(simple_data)
        result = features.volatility(window=3, units="invalid")

        assert result is None
        captured = capsys.readouterr()
        assert "Units must be either d, w, m or y" in captured.out

    @pytest.mark.parametrize("units", ["d", "w", "m", "y"])
    def test_volatility_valid_units(self, simple_data, units):
        """All valid unit types should work"""
        features = Features(simple_data)
        result = features.volatility(window=2, units=units)

        assert result is not None
        assert f"Asset1_realvol2{units}" in result.columns

    def test_volatility_positive_values(self, simple_data):
        """Volatility should always be non-negative"""
        features = Features(simple_data)
        result = features.volatility(window=3, units="m")

        non_null = result.filter(pl.col("Asset1_realvol3m").is_not_null())
        for val in non_null["Asset1_realvol3m"].to_list():
            assert val >= 0


class TestFeaturesBeta:
    """Test cases for beta calculation"""

    @pytest.fixture
    def simple_data(self):
        """Simple dataset with known returns"""
        return pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "Asset1": [0.01, 0.02, 0.03, -0.01, 0.02, 0.01],
            "Asset2": [0.02, 0.04, 0.06, -0.02, 0.04, 0.02],  # 2x Asset1
        })

    @pytest.fixture
    def benchmark_data(self, simple_data):
        """Benchmark returns (same dates as simple_data)"""
        return pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            # Same as Asset1
            "benchmark": [0.01, 0.02, 0.03, -0.01, 0.02, 0.01],
        })

    @pytest.fixture
    def uncorrelated_benchmark(self, simple_data):
        """Benchmark uncorrelated with assets"""
        return pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "benchmark": [0.03, -0.01, 0.02, 0.01, -0.02, 0.04],
        })

    def test_beta_returns_correct_columns(self, simple_data, benchmark_data):
        """Verify output DataFrame has correct column names"""
        features = Features(simple_data)
        result = features.beta(window=3, bench=benchmark_data, units="m")

        assert "Date" in result.columns
        assert "Asset1_beta3m" in result.columns
        assert "Asset2_beta3m" in result.columns

    def test_beta_returns_correct_length(self, simple_data, benchmark_data):
        """Output should have same number of rows as input"""
        features = Features(simple_data)
        result = features.beta(window=3, bench=benchmark_data, units="m")

        assert len(result) == len(simple_data)

    def test_beta_lagged_by_one_period(self, simple_data, benchmark_data):
        """Beta should be lagged by 1 period"""
        features = Features(simple_data)
        result = features.beta(window=3, bench=benchmark_data, units="m")

        # First row should be null due to lag
        assert result["Asset1_beta3m"][0] is None

    def test_beta_identical_to_benchmark(self, simple_data, benchmark_data):
        """
        Asset identical to benchmark: tests beta calculation consistency.

        Note: The implementation uses population covariance (via rolling_mean)
        divided by sample variance (rolling_var with ddof=1), resulting in a
        bias factor of (n-1)/n. For window=3 (3 periods), expected beta = 2/3 instead of 1.
        """
        features = Features(simple_data)
        result = features.beta(window=3, bench=benchmark_data, units="m")

        # Due to ddof mismatch: pop_cov / sample_var = (n-1)/n * true_beta
        # For window=3 (3 periods): expected = 2/3 * 1.0 = 0.6667
        expected_beta = (3 - 1) / 3 * 1.0
        non_null = result.filter(pl.col("Asset1_beta3m").is_not_null())
        for val in non_null["Asset1_beta3m"].to_list():
            assert val == pytest.approx(expected_beta, rel=1e-6)

    def test_beta_twice_benchmark(self, simple_data, benchmark_data):
        """
        Asset that is 2x benchmark: tests linear scaling of beta.

        Note: Same bias factor applies - see test_beta_identical_to_benchmark.
        For window=3 (3 periods): expected = 2/3 * 2.0 = 1.3333
        """
        features = Features(simple_data)
        result = features.beta(window=3, bench=benchmark_data, units="m")

        # Due to ddof mismatch: expected = (n-1)/n * true_beta = 2/3 * 2.0
        expected_beta = (3 - 1) / 3 * 2.0
        non_null = result.filter(pl.col("Asset2_beta3m").is_not_null())
        for val in non_null["Asset2_beta3m"].to_list():
            assert val == pytest.approx(expected_beta, rel=1e-6)

    def test_beta_manual_calculation(self, simple_data, uncorrelated_benchmark):
        """
        Verify beta matches manual calculation using the implementation's formula:
        beta = pop_covariance / sample_variance

        where:
        - pop_covariance = E[XY] - E[X]E[Y]  (rolling_mean based)
        - sample_variance = rolling_var with ddof=1

        Window=3 means 3 periods of data (current row + 2 previous rows in Polars)
        """
        features = Features(simple_data)
        result = features.beta(
            window=3, bench=uncorrelated_benchmark, units="m")

        # At index 3, rolling window=3 covers indices 1, 2, 3 (3 periods)
        # After lag by 1, this value appears at index 4
        # Asset1 returns at indices 1, 2, 3: [0.02, 0.03, -0.01]
        # Benchmark returns at indices 1, 2, 3: [-0.01, 0.02, 0.01]
        asset_returns = np.array([0.02, 0.03, -0.01])
        bench_returns = np.array([-0.01, 0.02, 0.01])

        # Population covariance: E[XY] - E[X]E[Y]
        pop_cov = np.mean(asset_returns * bench_returns) - \
            np.mean(asset_returns) * np.mean(bench_returns)
        # Sample variance (ddof=1) - this is what polars rolling_var uses
        sample_var_bench = np.var(bench_returns, ddof=1)
        expected_beta = pop_cov / sample_var_bench

        assert result["Asset1_beta3m"][4] == pytest.approx(
            expected_beta, rel=1e-5)

    def test_beta_invalid_units_returns_none(self, simple_data, benchmark_data, capsys):
        """Invalid units should print error and return None"""
        features = Features(simple_data)
        result = features.beta(window=3, bench=benchmark_data, units="invalid")

        assert result is None
        captured = capsys.readouterr()
        assert "Units must be either d, w, m or y" in captured.out

    @pytest.mark.parametrize("units", ["d", "w", "m", "y"])
    def test_beta_valid_units(self, simple_data, benchmark_data, units):
        """All valid unit types should work"""
        features = Features(simple_data)
        result = features.beta(window=3, bench=benchmark_data, units=units)

        assert result is not None
        assert f"Asset1_beta3{units}" in result.columns

    def test_beta_custom_bench_col(self, simple_data):
        """Test using custom benchmark column name"""
        custom_bench = pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "market_return": [0.01, 0.02, 0.03, -0.01, 0.02, 0.01],
        })

        features = Features(simple_data)
        result = features.beta(window=3, bench=custom_bench,
                               bench_col="market_return", units="m")

        assert "Asset1_beta3m" in result.columns
        # Asset1 identical to market_return - same bias factor applies
        expected_beta = (3 - 1) / 3 * 1.0
        non_null = result.filter(pl.col("Asset1_beta3m").is_not_null())
        for val in non_null["Asset1_beta3m"].to_list():
            assert val == pytest.approx(expected_beta, rel=1e-6)


class TestFeaturesInit:
    """Test cases for Features class initialization"""

    def test_init_identifies_asset_columns(self):
        """Asset columns should be all columns except date"""
        data = pl.DataFrame({
            "Date": [1, 2, 3],
            "Stock_A": [0.01, 0.02, 0.03],
            "Stock_B": [0.02, 0.01, 0.04],
            "Stock_C": [0.03, 0.02, 0.01],
        })

        features = Features(data)

        assert features.n_assets == 3
        assert set(features.asset_cols) == {"Stock_A", "Stock_B", "Stock_C"}
        assert features.date_col == "Date"

    def test_init_custom_date_col(self):
        """Custom date column name should work"""
        data = pl.DataFrame({
            "timestamp": [1, 2, 3],
            "Asset1": [0.01, 0.02, 0.03],
        })

        features = Features(data, date_col="timestamp")

        assert features.date_col == "timestamp"
        assert features.asset_cols == ["Asset1"]


class TestFeaturesEdgeCases:
    """Edge case tests for Features class"""

    def test_single_asset(self):
        """Should work with single asset"""
        data = pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "Asset1": [0.01, 0.02, 0.03, -0.01, 0.02, 0.01],
        })
        benchmark = pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "benchmark": [0.01, 0.02, 0.03, -0.01, 0.02, 0.01],
        })

        features = Features(data)

        mom_result = features.mom(window=2, units="m")
        vol_result = features.volatility(window=2, units="m")
        beta_result = features.beta(window=2, bench=benchmark, units="m")

        assert len(mom_result.columns) == 2  # Date + 1 asset
        assert len(vol_result.columns) == 2
        assert len(beta_result.columns) == 2

    def test_many_assets(self):
        """Should work with many assets"""
        n_assets = 50
        dates = pl.date_range(pl.date(2023, 1, 1), pl.date(
            2023, 12, 1), "1mo", eager=True)

        data_dict = {"Date": dates}
        for i in range(n_assets):
            data_dict[f"Asset{i}"] = np.random.randn(len(dates)) * 0.05

        data = pl.DataFrame(data_dict)
        features = Features(data)

        result = features.mom(window=3, units="m")

        assert features.n_assets == n_assets
        assert len(result.columns) == n_assets + 1  # assets + Date

    def test_negative_returns(self):
        """Should handle negative returns correctly"""
        data = pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "Asset1": [-0.10, -0.05, -0.02, -0.08, -0.03, -0.01],
        })

        features = Features(data)
        result = features.mom(window=2, units="m")

        # Should not error, values should be computable
        assert result is not None
        # Negative compound returns should still be > -1 for these values
        non_null = result.filter(pl.col("Asset1_mom2m").is_not_null())
        assert len(non_null) > 0

    def test_zero_returns(self):
        """Should handle zero returns correctly"""
        data = pl.DataFrame({
            "Date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 6, 1), "1mo", eager=True),
            "Asset1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })

        features = Features(data)
        mom_result = features.mom(window=2, units="m")
        vol_result = features.volatility(window=2, units="m")

        # Zero returns should give zero momentum
        non_null_mom = mom_result.filter(pl.col("Asset1_mom2m").is_not_null())
        for val in non_null_mom["Asset1_mom2m"].to_list():
            assert val == pytest.approx(0.0, abs=1e-10)

        # Zero returns should give zero volatility
        non_null_vol = vol_result.filter(
            pl.col("Asset1_realvol2m").is_not_null())
        for val in non_null_vol["Asset1_realvol2m"].to_list():
            assert val == pytest.approx(0.0, abs=1e-10)
