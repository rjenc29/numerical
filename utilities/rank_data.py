import numpy as np
from numba import njit


@njit
def rank_data(A):
    """
    Rank supplied data column-wise.  Ties are dealt with using
    the average method whereby the average of the ranks that would
    have been assigned to all the tied values is assigned to each value.
    """
    assert A.ndim > 1

    A = A.astype(np.float64)  # may result in spurious ties or differences across platforms

    res = np.empty_like(A)
    m = A.shape[0]

    for i in range(A.shape[1]):
        data_i = A[:, i]
        data_i_std = np.empty_like(data_i, dtype=np.int32)
        sort_order = np.argsort(data_i)

        data_i = data_i[sort_order]
        obs = np.ones(m, dtype=np.int32)

        for j in range(m):
            idx = sort_order[j]
            data_i_std[idx] = j
            if j > 0:
                if data_i[j] == data_i[j - 1]:
                    obs[j] = 0

        dense = obs.cumsum()[data_i_std]

        non_zero_indices = np.nonzero(obs)[0]
        count = np.empty(len(non_zero_indices) + 1)
        count[:-1] = non_zero_indices
        count[-1] = m

        res[:, i] = (count[dense] + count[dense - 1] + 1) / 2.0

    return res
