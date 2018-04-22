import math

import numpy as np
from numba import types
from numba.extending import overload, register_jitable


@register_jitable
def _collect_percentiles(a, q, skip_nan=False):

    if np.any(np.isnan(q)):
        raise ValueError('q must not be NaN')

    a_sorted = np.sort(a.flatten())

    if skip_nan:
        nan_mask = np.isnan(a_sorted)
        a_sorted = a_sorted[~nan_mask]
        if len(a_sorted) == 0:
            return np.full(len(q), np.nan)
    else:
        if np.any(np.isnan(a_sorted)):
            return np.full(len(q), np.nan)

    out = np.empty(len(q))
    i = 0

    for v in np.nditer(q):
        percentile = v.item()
        if percentile < 0 or percentile > 100:
            raise ValueError("Percentiles must be in the range [0,100]")

        if percentile == 0:
            val = a_sorted[0]
        elif percentile == 100:
            val = a_sorted[-1]
        else:
            rank = percentile / 100 * (len(a_sorted) - 1) + 1
            f = math.floor(rank)
            m = rank - f
            val = a_sorted[f - 1] + m * (a_sorted[f] - a_sorted[f - 1])
        out[i] = val
        i += 1

    return out


@overload(np.percentile)
def np_percentile(a, q):

    def np_percentile_q_scalar_impl(a, q):
        q = np.array([q])
        return _collect_percentiles(a, q)[0]

    def np_percentile_q_array_impl(a, q):
        return _collect_percentiles(a, q)

    def np_percentile_q_tuple_impl(a, q):
        q = np.array(q)
        return _collect_percentiles(a, q)

    if isinstance(q, (types.Float, types.Integer)):
        fn = np_percentile_q_scalar_impl

    elif isinstance(q, (types.Tuple)):
        fn = np_percentile_q_tuple_impl

    elif isinstance(q, types.Array):
        fn = np_percentile_q_array_impl

    else:
        raise ValueError('q must be scalar, tuple or np.array')

    return fn


@overload(np.nanpercentile)
def np_nanpercentile(a, q):

    def np_nanpercentile_q_scalar_impl(a, q):
        q = np.array([q])
        return _collect_percentiles(a, q, skip_nan=True)[0]

    def np_nanpercentile_q_array_impl(a, q):
        return _collect_percentiles(a, q, skip_nan=True)

    def np_nanpercentile_q_tuple_impl(a, q):
        q = np.array(q)
        return _collect_percentiles(a, q, skip_nan=True)

    if isinstance(q, (types.Float, types.Integer)):
        fn = np_nanpercentile_q_scalar_impl

    elif isinstance(q, (types.Tuple)):
        fn = np_nanpercentile_q_tuple_impl

    elif isinstance(q, types.Array):
        fn = np_nanpercentile_q_array_impl

    else:
        raise ValueError('q must be scalar, tuple or np.array')

    return fn
