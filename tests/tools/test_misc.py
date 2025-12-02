"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from droplets.tools import misc


def test_enable_scalar_args():
    """Test the enable_scalar_args decorator."""

    class Test:
        @misc.enable_scalar_args
        def meth(self, arr):
            return arr + 1

    t = Test()

    assert t.meth(1) == 2
    assert isinstance(t.meth(1), float)
    assert isinstance(t.meth(1.0), float)
    np.testing.assert_equal(t.meth(np.ones(2)), np.full(2, 2))
