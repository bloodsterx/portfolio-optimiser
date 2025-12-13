import pytest
import numpy as np
from src.data.covariance import compute_rolling_covariance


class TestRollingCovariance:
    """Test cases for rolling covariance estimation"""

    @pytest.fixture
    def simple_returns(self):
        """Simple returns data with known values"""
        return np.array([
            [0.01, 0.02],
            [0.02, 0.04],
            [0.03, 0.06],
            [-0.01, -0.02],
            [0.02, 0.04],
            [0.01, 0.02],
        ])

    @pytest.fixture
    def perfectly_correlated_returns(self):
        """Asset2 = 2 * Asset1 (perfect correlation)"""
        asset1 = np.array([0.01, 0.02, 0.03, -0.01, 0.02, 0.01])
        asset2 = 2 * asset1
        return np.column_stack([asset1, asset2])

    def test_output_shape(self, simple_returns):
        """Output should have shape (T, n_assets, n_assets)"""
        result = compute_rolling_covariance(simple_returns, window=3, method='sample')
        
        T, n_assets = simple_returns.shape
        assert result.shape == (T, n_assets, n_assets)

    def test_warmup_period_zeros(self, simple_returns):
        """Early periods (t < window-1) should be zeros"""
        window = 3
        result = compute_rolling_covariance(simple_returns, window=window, method='sample')
        
        # First 2 matrices (indices 0, 1) should be all zeros
        for t in range(window - 1):
            assert np.allclose(result[t], np.zeros((2, 2)))

    def test_covariance_matrix_symmetry(self, simple_returns):
        """Covariance matrices should be symmetric"""
        result = compute_rolling_covariance(simple_returns, window=3, method='sample')
        
        for t in range(3, len(simple_returns)):  # Skip warmup
            assert np.allclose(result[t], result[t].T)

    def test_covariance_matrix_positive_diagonal(self, simple_returns):
        """Diagonal elements (variances) should be non-negative"""
        result = compute_rolling_covariance(simple_returns, window=3, method='sample')
        
        for t in range(3, len(simple_returns)):  # Skip warmup
            assert np.all(np.diag(result[t]) >= 0)

    def test_sample_method_manual_calculation(self, simple_returns):
        """Verify sample covariance matches manual numpy calculation"""
        window = 3
        result = compute_rolling_covariance(simple_returns, window=window, method='sample')
        
        # At t=2 (first valid), window covers indices 0, 1, 2
        window_data = simple_returns[0:3]
        expected_cov = np.cov(window_data, rowvar=False) + np.eye(2) * 1e-6
        
        assert np.allclose(result[2], expected_cov, rtol=1e-6)

    def test_ledoit_wolf_method(self, simple_returns):
        """Ledoit-Wolf method should produce valid covariance matrices"""
        result = compute_rolling_covariance(simple_returns, window=3, method='ledoit_wolf')
        
        # Check shape and basic properties
        assert result.shape == (6, 2, 2)
        
        # Check symmetry and positive diagonal for valid periods
        for t in range(2, len(simple_returns)):
            assert np.allclose(result[t], result[t].T)
            assert np.all(np.diag(result[t]) >= 0)

    def test_rolling_window_moves(self, simple_returns):
        """Verify the rolling window actually moves"""
        window = 3
        result = compute_rolling_covariance(simple_returns, window=window, method='sample')
        
        # t=2 uses indices [0, 1, 2]
        # t=3 uses indices [1, 2, 3]
        # These should produce different covariance matrices
        assert not np.allclose(result[2], result[3])

    def test_perfectly_correlated_assets(self, perfectly_correlated_returns):
        """Perfectly correlated assets should have correlation close to 1
        
        Note: Due to the small ridge regularization (1e-6) added to diagonal,
        correlation will be slightly less than 1.0
        """
        result = compute_rolling_covariance(
            perfectly_correlated_returns, window=3, method='sample'
        )
        
        # Check correlation at t=2 (first valid period)
        cov_matrix = result[2]
        # Correlation = cov(X,Y) / (std(X) * std(Y))
        corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        # Allow small deviation due to ridge regularization on diagonal
        assert corr == pytest.approx(1.0, rel=1e-2)

    def test_window_1_edge_case(self):
        """Window of 1 should use single observation (variance undefined for sample)"""
        returns = np.array([[0.01, 0.02], [0.03, 0.04]])
        result = compute_rolling_covariance(returns, window=1, method='sample')
        
        # With window=1, we get a single observation, np.cov returns scalar for single row
        # This is an edge case - the result depends on numpy's behavior
        assert result.shape == (2, 2, 2)

    def test_larger_window_than_data(self):
        """Window larger than data should produce all zeros"""
        returns = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
        result = compute_rolling_covariance(returns, window=10, method='sample')
        
        # All matrices should be zeros since we never have enough data
        assert np.allclose(result, np.zeros((3, 2, 2)))

