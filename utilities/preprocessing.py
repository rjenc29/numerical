import numpy as np
from numba import jit


@jit(nopython=True)
def standard_scale(data):

    n = data.shape[1]

    res = np.empty_like(data)

    data_t = data.T

    for i in range(n):
        data_i = data_t[i, :]
        res[:, i] = (data_i - np.mean(data_i)) / np.std(data_i)

    return res


if __name__ == '__main__':
    import pytest
    pytest.main()
