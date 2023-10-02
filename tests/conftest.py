"""
This file is used to configure the test environment when running py.test

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pde.tools.numba import random_seed


@pytest.fixture(scope="function", autouse=False, name="rng")
def init_random_number_generators():
    """get a random number generator and set the seed of the random number generator

    The function returns an instance of :func:`~numpy.random.default_rng()` and
    initializes the default generators of both :mod:`numpy` and :mod:`numba`.
    """
    random_seed()
    return np.random.default_rng(0)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """helper function adjusting environment before and after tests"""
    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test
    yield

    # clean up open matplotlib figures after the test
    plt.close("all")
