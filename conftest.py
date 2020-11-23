import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def enable_raise_errors():
    """ helper function enabling errors for all tests """
    # prepare something ahead of all tests
    np.seterr(all="raise", under="ignore")
