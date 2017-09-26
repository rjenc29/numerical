import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from utilities.preprocessing import standard_scale


def test_standard_scale():
    predictors = load_iris().data
    scaler = StandardScaler()

    expected = scaler.fit_transform(predictors)
    output = standard_scale(predictors)
    assert np.allclose(expected, output)


if __name__ == '__main__':
    import pytest
    pytest.main()
