import numpy as np
from scipy.stats import rankdata
from sklearn.datasets import load_iris

from utilities.rank_data import rank_data


def test_rank_data():
    data = load_iris().data

    # rank the data all at once
    output = rank_data(data)

    # check each column versus scipy equivalent
    for i in range(data.shape[1]):
        feature = data[:, i]
        expected = rankdata(feature)
        assert np.allclose(expected, output[:, i])


if __name__ == '__main__':
    import pytest
    pytest.main()
