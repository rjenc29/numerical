import pandas as pd
import numpy as np

from utilities.rolling_stats import rolling_sum
from pytest import raises


def test_rolling_sum_series_containing_nans():
    x = np.arange(30).astype(float)
    s = pd.Series(x)
    s[0] = np.nan
    s[6] = np.nan
    s[12:18] = np.nan
    s[-1] = np.nan

    window = 3

    expected = s.rolling(window=window).sum().values
    output = rolling_sum(s.values, window)
    assert np.allclose(output, expected, equal_nan=True)


def test_rolling_sum_no_nans():
    x = np.arange(100).astype(float)
    s = pd.Series(x)

    window = 5

    expected = s.rolling(window=window).sum().values
    output = rolling_sum(s.values, window)
    assert np.allclose(output, expected, equal_nan=True)


def test_rolling_sum_various_window_sizes():
    x = np.arange(40).astype(float)
    s = pd.Series(x)
    s[:10] = np.nan

    for window in 1, 2, 5, 10, 30, 50:
        expected = s.rolling(window=window).sum().values
        output = rolling_sum(s.values, window)
        assert np.allclose(output, expected, equal_nan=True)


def test_rolling_sum_window_size_one():
    x = np.arange(10).astype(float)
    s = pd.Series(x)
    window = 1

    expected = s.rolling(window=window).sum().values
    output = rolling_sum(s.values, window)
    assert np.allclose(output, expected)
    assert np.allclose(output, s.values)


def test_rolling_sum_window_size_zero():
    x = np.arange(10).astype(float)
    s = pd.Series(x)
    window = 0

    output = rolling_sum(s.values, window)
    assert all(np.isnan(output))


def test_rolling_sum_window_size_negative():
    x = np.arange(10).astype(float)
    s = pd.Series(x)
    window = -1

    with raises(ValueError, message="window must be non-negative"):
        _ = rolling_sum(s.values, window)


if __name__ == '__main__':
    import pytest
    pytest.main()
