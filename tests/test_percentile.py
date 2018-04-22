
import numpy as np
from numba import njit
from numpy.testing import assert_allclose
from pytest import approx, raises

import utilities.percentile as perc

_ = perc  # to prevent it appearing to be an unused import


@njit
def np_percentile_jit(x, q):
    return np.percentile(x, q)


@njit
def np_nanpercentile_jit(x, q):
    return np.nanpercentile(x, q)


def test_scalar_q():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2])
    q = 2.2

    output = np_percentile_jit(arr, q)
    expected = np.percentile(arr, q)
    assert output == approx(expected)


def test_tuple_q():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2])
    q = (2.2, 34, 4)

    output = np_percentile_jit(arr, q)
    expected = np.percentile(arr, q)
    assert output == approx(expected)


def test_array_q():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2])
    q = np.array([0, 2.2, 34, 100])

    output = np_percentile_jit(arr, q)
    expected = np.percentile(arr, q)
    assert output == approx(expected)


def test_array_q_contains_nan():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2])
    q = np.array([0, 2.2, np.nan, 100])

    with raises(ValueError):
        _ = np.percentile(arr, q)

    with raises(ValueError):
        _ = np_percentile_jit(arr, q)


def test_array_arr_contains_nan():

    arr = np.array([1, 4, 3, 3.3, 6, np.nan, 2.2])
    q = np.array([0, 2.2, 100])

    output = np.percentile(arr, q)
    assert len(output) == 3
    assert np.all(np.isnan(output))

    output = np_percentile_jit(arr, q)
    assert len(output) == 3
    assert np.all(np.isnan(output))


def test_multi_dimensional_array():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2, 2.3, 0]).reshape(3, 3)
    q = np.array([0, 2.2, 34, 100])

    output = np_percentile_jit(arr, q)
    expected = np.percentile(arr, q)
    assert_allclose(output, expected)


######### nan percentile

def test_nanpercentile_scalar_q():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2])
    q = 2.2

    output = np_nanpercentile_jit(arr, q)
    expected = np.nanpercentile(arr, q)
    assert output == approx(expected)


def test_nanpercentile_tuple_q():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2])
    q = (2.2, 34, 4)

    output = np_nanpercentile_jit(arr, q)
    expected = np.nanpercentile(arr, q)
    assert output == approx(expected)


def test_nanpercentile_array_q():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2])
    q = np.array([0, 2.2, 34, 100])

    output = np_nanpercentile_jit(arr, q)
    expected = np.nanpercentile(arr, q)
    assert output == approx(expected)


def test_nanpercentile_array_q_contains_nan():

    arr = np.array([1, 4, 3, 3.3, 6, 5, 2.2])
    q = np.array([0, 2.2, np.nan, 100])

    with raises(ValueError):
        _ = np.nanpercentile(arr, q)

    with raises(ValueError):
        _ = np_nanpercentile_jit(arr, q)


def test_nanpercentile_arr_contains_nan():

    arr = np.array([1, 4, 3, 3.3, 6, np.nan, 2.2])
    q = np.array([0, 2.2, 100])

    expected = np.nanpercentile(arr, q)
    output = np_nanpercentile_jit(arr, q)
    assert_allclose(output, expected)


def test_nanpercentile_multi_dimensional_array():

    arr = np.array([1, 4, np.nan, 3.3, 6, 5, 2.2, 2.3, 0]).reshape(3, 3)
    q = np.array([0, 2.2, 34, 100])

    output = np_nanpercentile_jit(arr, q)
    expected = np.nanpercentile(arr, q)
    assert_allclose(output, expected)


def test_nanpercentile_arr_all_but_one_nan():

    arr = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3.3, np.nan])
    q = 17.7

    expected = np.nanpercentile(arr, q)
    output = np_nanpercentile_jit(arr, q)
    assert_allclose(output, expected)


if __name__ == '__main__':
    import pytest
    pytest.main()
