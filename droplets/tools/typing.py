"""Miscellaneous types.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.integer | np.floating]
