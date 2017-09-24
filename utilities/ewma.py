import numpy as np
from numpy import empty


def ewma(data, alpha, adjust, ignore_na):

    old_wt_factor = 1. - alpha
    new_wt = 1. if adjust else alpha

    n = data.shape[0]
    output = empty(n)

    weighted_avg = data[0]
    is_observation = (weighted_avg == weighted_avg)
    nobs = int(is_observation)
    output[0] = weighted_avg if (nobs >= 1) else np.nan
    old_wt = 1.

    for i in range(1, n):
        cur = data[i]
        is_observation = (cur == cur)
        nobs += int(is_observation)
        if weighted_avg == weighted_avg:

            if is_observation or (not ignore_na):

                old_wt *= old_wt_factor
                if is_observation:

                    if weighted_avg != cur:
                        weighted_avg = ((old_wt * weighted_avg) +
                                        (new_wt * cur)) / (old_wt + new_wt)
                    if adjust:
                        old_wt += new_wt
                    else:
                        old_wt = 1.
        elif is_observation:
            weighted_avg = cur

        output[i] = weighted_avg if (nobs >= 1) else np.nan

    return output


if __name__ == '__main__':
    import pytest
    pytest.main()
