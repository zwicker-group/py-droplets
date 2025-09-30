"""Miscellaneous types.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[Union[np.integer, np.floating]]
