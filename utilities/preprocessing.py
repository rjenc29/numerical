import numpy as np
from numba import jit


@jit(nopython=True)
def standard_scale(data, ddof=0):

    if ddof not in (0, 1):
        raise ValueError('ddof must be either 0 or 1')

    n = data.shape[1]
    res = np.empty_like(data)
    data_t = data.T

    for i in range(n):
        data_i = data_t[i, :]
        res[:, i] = (data_i - np.mean(data_i)) / np.std(data_i)

    if ddof == 0:
        return res
    elif ddof == 1:
        m = data.shape[0]
        return res * np.sqrt((m - 1) / m)


@jit(nopython=True)
def min_max_scale(data):

    n = data.shape[1]
    res = np.empty_like(data)
    data_t = data.T

    for i in range(n):
        data_i = data_t[i, :]
        res[:, i] = (data_i - np.min(data_i)) / (np.max(data_i) - np.min(data_i))

    return res


if __name__ == '__main__':
    import pytest
    pytest.main()
