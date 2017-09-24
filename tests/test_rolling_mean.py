import pandas as pd
import numpy as np

from utilities.rolling_mean import rolling_mean
from pytest import raises


def test_rolling_mean_series_containing_nans():

    # set up some sample test data containing some NaNs
    x = np.arange(30).astype(float)
    s = pd.Series(x)
    s[0] = np.nan
    s[6] = np.nan
    s[12:18] = np.nan
    s[-1] = np.nan

    # define a small, arbitrary window
    window = 3

    # check outputs
    expected = s.rolling(window=window).mean().values
    output = rolling_mean(s.values, window)
    assert np.allclose(output, expected, equal_nan=True)


def test_rolling_mean_no_nans():

    # set up some sample test data containing some NaNs
    x = np.arange(100).astype(float)
    s = pd.Series(x)

    # define a small, arbitrary window
    window = 5

    # check outputs
    expected = s.rolling(window=window).mean().values
    output = rolling_mean(s.values, window)
    assert np.allclose(output, expected, equal_nan=True)


def test_rolling_mean_various_window_sizes():

    # set up some sample test data containing some NaNs
    x = np.arange(40).astype(float)
    s = pd.Series(x)
    s[:10] = np.nan

    for window in 1, 2, 5, 10, 30, 50:
        expected = s.rolling(window=window).mean().values
        output = rolling_mean(s.values, window)
        assert np.allclose(output, expected, equal_nan=True)


def test_rolling_mean_window_size_one():

    # set up some sample test data containing some NaNs
    x = np.arange(10).astype(float)
    s = pd.Series(x)
    window = 1

    expected = s.rolling(window=window).mean().values
    output = rolling_mean(s.values, window)
    assert np.allclose(output, expected)
    assert np.allclose(output, s.values)


def test_rolling_mean_window_size_zero():

    # set up some sample test data containing some NaNs
    x = np.arange(10).astype(float)
    s = pd.Series(x)
    window = 0

    output = rolling_mean(s.values, window)
    assert all(np.isnan(output))


def test_rolling_mean_window_size_negative():

    # set up some sample test data containing some NaNs
    x = np.arange(10).astype(float)
    s = pd.Series(x)
    window = -1

    with raises(ValueError, message="window must be non-negative"):
        _ = rolling_mean(s.values, window)


if __name__ == '__main__':
    import pytest
    pytest.main()
