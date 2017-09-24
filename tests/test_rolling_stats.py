import pandas as pd
import numpy as np

from unittest import TestCase
from utilities.rolling_stats import rolling_sum, rolling_mean
from pytest import raises


class RollingStatsTests:

    def test_series_containing_nans(self):
        x = np.arange(30).astype(float)
        s = pd.Series(x)
        s[0] = np.nan
        s[6] = np.nan
        s[12:18] = np.nan
        s[-1] = np.nan

        window = 3

        expected = getattr(s.rolling(window=window), self._pandas_func_name)().values
        output = self._func(s.values, window)
        assert np.allclose(output, expected, equal_nan=True)

    def test_series_with_no_nans(self):
        x = np.arange(100).astype(float)
        s = pd.Series(x)

        window = 5

        expected = getattr(s.rolling(window=window), self._pandas_func_name)().values
        output = self._func(s.values, window)
        assert np.allclose(output, expected, equal_nan=True)

    def test_series_various_window_sizes(self):
        x = np.arange(40).astype(float)
        s = pd.Series(x)
        s[:10] = np.nan

        for window in 1, 2, 5, 10, 30, 50:
            expected = getattr(s.rolling(window=window), self._pandas_func_name)().values
            output = self._func(s.values, window)
            assert np.allclose(output, expected, equal_nan=True)

    def test_series_window_size_one(self):
        x = np.arange(10).astype(float)
        s = pd.Series(x)
        window = 1

        expected = getattr(s.rolling(window=window), self._pandas_func_name)().values
        output = self._func(s.values, window)
        assert np.allclose(output, expected)
        assert np.allclose(output, s.values)

    def test_series_window_size_zero(self):
        x = np.arange(10).astype(float)
        s = pd.Series(x)
        window = 0

        output = self._func(s.values, window)
        assert all(np.isnan(output))

    def test_series_window_size_negative(self):
        x = np.arange(10).astype(float)
        s = pd.Series(x)
        window = -1

        with raises(ValueError, message="window must be non-negative"):
            _ = self._func(s.values, window)


class RollingMeanTests(TestCase, RollingStatsTests):

    def setUp(self):
        super().setUp()
        self._func = rolling_mean
        self._pandas_func_name = 'mean'


class RollingSumTests(TestCase, RollingStatsTests):

    def setUp(self):
        super().setUp()
        self._func = rolling_sum
        self._pandas_func_name = 'sum'


if __name__ == '__main__':
    import pytest
    pytest.main()
