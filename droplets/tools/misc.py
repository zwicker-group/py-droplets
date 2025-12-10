"""Miscellaneous functions.

.. autosummary::
   :nosignatures:

   enable_scalar_args

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from pde.tools.misc import number_array

TFunc = TypeVar("TFunc", bound=Callable[..., Any])


def enable_scalar_args(method: TFunc) -> TFunc:
    """Decorator that makes vectorized methods work with scalars.

    This decorator allows to call functions that are written to work on numpy arrays to
    also accept python scalars, like `int` and `float`. Essentially, this wrapper turns
    them into an array and unboxes the result. Note that the dtype of the returned value
    will always be double or cdouble even if the function is called with an integer.

    Args:
        method: The method being decorated

    Returns:
        The decorated method
    """

    @functools.wraps(method)
    def wrapper(self, *args):
        args = [number_array(arg, copy=None) for arg in args]
        if args[0].ndim == 0:
            args = [arg[None] for arg in args]
            return method(self, *args)[0]
        return method(self, *args)

    return wrapper  # type: ignore
