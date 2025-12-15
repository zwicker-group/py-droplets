"""This file is used to configure the test environment when running py.test.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pde.backends.numba.utils import random_seed


@pytest.fixture(autouse=False, name="rng")
def init_random_number_generators():
    """Get a random number generator and set the seed of the random number generator.

    The function returns an instance of :func:`~numpy.random.default_rng()` and
    initializes the default generators of both :mod:`numpy` and :mod:`numba`.
    """
    random_seed()
    return np.random.default_rng(0)


@pytest.fixture(autouse=True)
def _setup_and_teardown():
    """Helper function adjusting environment before and after tests."""
    # ensure we use the Agg backend, so figures are not displayed
    plt.switch_backend("agg")
    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test
    yield

    # clean up open matplotlib figures after the test
    plt.close("all")
