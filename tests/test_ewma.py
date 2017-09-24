import pandas as pd
import numpy as np

from utilities.ewma import ewma


def sample_data_series():
    data = np.arange(50).astype(float)
    data[3] = np.nan
    data[4] = np.nan
    return pd.Series(data)


def test_ewma_adjust_and_ignore_na():
    series = sample_data_series()
    alpha = 0.1

    expected = series.ewm(alpha=alpha, adjust=True, ignore_na=True).mean()
    output = ewma(series.values, alpha, True, True)

    assert np.allclose(expected, output)


def test_ewma_adjust():
    series = sample_data_series()
    alpha = 0.1

    expected = series.ewm(alpha=alpha, adjust=True, ignore_na=False).mean()
    output = ewma(series.values, alpha, True, False)

    assert np.allclose(expected, output)


def test_ewma_ignore_na():
    series = sample_data_series()
    alpha = 0.1

    expected = series.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    output = ewma(series.values, alpha, False, True)

    assert np.allclose(expected, output)


def test_ewma():
    series = sample_data_series()
    alpha = 0.1

    expected = series.ewm(alpha=alpha, adjust=False, ignore_na=False).mean()
    output = ewma(series.values, alpha, False, False)

    assert np.allclose(expected, output)


if __name__ == '__main__':
    import pytest
    pytest.main()
