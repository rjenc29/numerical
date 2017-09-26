import numpy as np
from unittest import TestCase
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
from utilities.preprocessing import standard_scale, min_max_scale


class RollingStatsTests:

    def test_scaled_values(self):
        predictors = load_iris().data
        scaler = self._scaler()

        expected = scaler.fit_transform(predictors)
        output = self._func(predictors)
        assert np.allclose(expected, output)


class StandardScaleTests(TestCase, RollingStatsTests):

    def setUp(self):
        super().setUp()
        self._func = standard_scale
        self._scaler = StandardScaler


class MinMaxScaleTests(TestCase, RollingStatsTests):

    def setUp(self):
        super().setUp()
        self._func = min_max_scale
        self._scaler = MinMaxScaler


if __name__ == '__main__':
    import pytest
    pytest.main()
