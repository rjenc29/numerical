import numpy as np

from numba import jit


@jit(nopython=True)
def _rolling_statistic(x, window, window_divisor):

    if window < 0:
        raise ValueError('window must be non-negative')

    n = x.shape[0]

    if window == 0:
        return np.full(n, np.nan)

    if window == 1:
        return x

    res = np.empty(n)

    _sum = 0
    nans_in_window = 0

    for i in range(n):
        data_i = x[i]
        _nan_arrived = np.isnan(data_i)

        if _nan_arrived:
            nans_in_window = min(window, nans_in_window + 1)
        else:
            _sum += data_i

        if i < window - 1:
            res[i] = np.nan
            continue

        if i == window - 1:
            if nans_in_window == 0:
                res[i] = _sum / window_divisor
                continue

        if i >= window:
            evict_i = x[i - window]
            _evict_nan = np.isnan(evict_i)

            if _evict_nan:
                if not _nan_arrived:
                    nans_in_window -= 1
            else:
                _sum -= evict_i

            if nans_in_window == 0:
                res[i] = _sum / window_divisor
                continue

        res[i] = np.nan

    return res


def rolling_sum(x, window):
    return _rolling_statistic(x, window, 1)


def rolling_mean(x, window):
    return _rolling_statistic(x, window, window)


if __name__ == '__main__':
    import pytest
    pytest.main()
