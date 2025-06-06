r"""Module collecting functions for handling spherical geometry.

The coordinate systems use the following convention for polar coordinates
:math:`(r, \phi)`, where :math:`r` is the radial coordinate and :math:`\phi` is
the polar angle:

.. math::
    \begin{cases}
        x = r \cos(\phi) &\\
        y = r \sin(\phi) &
    \end{cases}
    \text{for} \; r \in [0, \infty] \;
    \text{and} \; \phi \in [0, 2\pi)

Similarly, for spherical coordinates :math:`(r, \theta, \phi)`, where :math:`r`
is the radial coordinate, :math:`\theta` is the azimuthal angle, and
:math:`\phi` is the polar angle, we use

.. math::
    \begin{cases}
        x = r \sin(\theta) \cos(\phi) &\\
        y = r \sin(\theta) \sin(\phi) &\\
        z = r \cos(\theta)
    \end{cases}
    \text{for} \; r \in [0, \infty], \;
    \theta \in [0, \pi], \; \text{and} \;
    \phi \in [0, 2\pi)


The module also provides functions for handling spherical harmonics.
These spherical harmonics are described by the degree :math:`l` and the order
:math:`m` or, alternatively, by the mode :math:`k`. The relation between these
values is

.. math::
    k = l(l + 1) + m

and

.. math::
    l &= \text{floor}(\sqrt{k}) \\
    m &= k - l(l + 1)

We will use these indices interchangeably, although the mode :math:`k` is
preferred internally. Note that we also consider axisymmetric spherical
harmonics, where the order is always zero and the degree :math:`l` and the mode
:math:`k` are thus identical.


.. autosummary::
   :nosignatures:

   radius_from_volume
   volume_from_radius
   surface_from_radius
   radius_from_surface
   make_radius_from_volume_compiled
   make_volume_from_radius_compiled
   make_surface_from_radius_compiled
   points_cartesian_to_spherical
   points_spherical_to_cartesian
   polar_coordinates
   spherical_index_k
   spherical_index_lm
   spherical_index_count
   spherical_index_count_optimal
   spherical_harmonic_symmetric
   spherical_harmonic_real
   spherical_harmonic_real_k

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import typing as t
from typing import Callable, Literal, TypeVar

import numpy as np
from numba.extending import overload, register_jitable

try:
    from scipy.special import sph_harm_y
except ImportError:
    # support scipy version below 1.15.0
    from scipy.special import sph_harm

    sph_harm_y = lambda n, m, theta, phi: sph_harm(m, n, phi, theta)

from pde.grids.base import DimensionError, GridBase
from pde.grids.spherical import volume_from_radius
from pde.tools.numba import jit

π = float(np.pi)

TNumArr = TypeVar("TNumArr", float, np.ndarray)


def radius_from_volume(volume: TNumArr, dim: int) -> TNumArr:
    """Return the radius of a sphere with a given volume.

    Args:
        volume (float or :class:`~numpy.ndarray`):
            Volume of the sphere
        dim (int):
            Dimension of the space

    Returns:
        float or :class:`~numpy.ndarray`: Radius of the sphere
    """
    if dim == 1:
        return volume / 2  # type: ignore
    elif dim == 2:
        return np.sqrt(volume / π)  # type: ignore
    elif dim == 3:
        return (3 * volume / (4 * π)) ** (1 / 3)  # type: ignore
    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")


def make_radius_from_volume_compiled(dim: int) -> Callable[[TNumArr], TNumArr]:
    """Return a function calculating the radius of a sphere with a given volume.

    Args:
        dim (int):
            Dimension of the space

    Returns:
        function: A function that takes a volume and returns the radius
    """
    if dim == 1:

        def radius_from_volume(volume: TNumArr) -> TNumArr:
            return volume / 2  # type: ignore

    elif dim == 2:

        def radius_from_volume(volume: TNumArr) -> TNumArr:
            return np.sqrt(volume / π)  # type: ignore

    elif dim == 3:

        def radius_from_volume(volume: TNumArr) -> TNumArr:
            return (3 * volume / (4 * π)) ** (1 / 3)  # type: ignore

    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")
    return jit(radius_from_volume)  # type: ignore


def make_radius_from_volume_nd_compiled() -> Callable[[TNumArr, int], TNumArr]:
    """Return a function calculating the radius of a sphere with a given volume.

    Returns:
        function: A function that calculate the radius from a volume and dimension
    """

    @register_jitable
    def radius_from_volume(volume: TNumArr, dim: int) -> TNumArr:
        if dim == 1:
            return volume / 2  # type: ignore
        elif dim == 2:
            return np.sqrt(volume / π)  # type: ignore
        elif dim == 3:
            return (3 * volume / (4 * π)) ** (1 / 3)  # type: ignore
        raise NotImplementedError

    return radius_from_volume  # type: ignore


def make_volume_from_radius_compiled(dim: int) -> Callable[[TNumArr], TNumArr]:
    """Return a function calculating the volume of a sphere with a given radius.

    Args:
        dim (int):
            Dimension of the space

    Returns:
        function: A function that takes a radius and returns the volume
    """
    if dim == 1:

        def volume_from_radius(radius: TNumArr) -> TNumArr:
            return 2 * radius

    elif dim == 2:

        def volume_from_radius(radius: TNumArr) -> TNumArr:
            return π * radius**2  # type: ignore

    elif dim == 3:

        def volume_from_radius(radius: TNumArr) -> TNumArr:
            return 4 * π / 3 * radius**3  # type: ignore

    else:
        raise NotImplementedError(f"Cannot calculate the volume in {dim} dimensions")
    return jit(volume_from_radius)  # type: ignore


def make_volume_from_radius_nd_compiled() -> Callable[[TNumArr, int], TNumArr]:
    """Return a function calculating the volume of a sphere with a given radius.

    Returns:
        function: A function that calculates the volume using a radius and dimension
    """

    @register_jitable
    def volume_from_radius_impl(radius: TNumArr, dim: int) -> TNumArr:
        if dim == 1:
            return 2 * radius
        elif dim == 2:
            return π * radius**2  # type: ignore
        elif dim == 3:
            return 4 * π / 3 * radius**3  # type: ignore
        raise NotImplementedError

    return volume_from_radius_impl  # type: ignore


def surface_from_radius(radius: TNumArr, dim: int) -> TNumArr:
    """Return the surface area of a sphere with a given radius.

    Args:
        radius (float or :class:`~numpy.ndarray`):
            Radius of the sphere
        dim (int):
            Dimension of the space

    Returns:
        float or :class:`~numpy.ndarray`: Surface area of the sphere
    """
    if dim == 1:
        if isinstance(radius, np.ndarray):
            return np.broadcast_to(2, radius.shape)  # type: ignore
        else:
            return 2
    elif dim == 2:
        return 2 * π * radius  # type: ignore
    elif dim == 3:
        return 4 * π * radius**2  # type: ignore
    else:
        raise NotImplementedError(
            f"Cannot calculate the surface area in {dim} dimensions"
        )


def radius_from_surface(surface: TNumArr, dim: int) -> TNumArr:
    """Return the radius of a sphere with a given surface area.

    Args:
        surface (float or :class:`~numpy.ndarray`):
            Surface area of the sphere
        dim (int):
            Dimension of the space

    Returns:
        float or :class:`~numpy.ndarray`: Radius of the sphere
    """
    if dim == 1:
        raise RuntimeError("Cannot calculate radius of 1-d sphere from surface")
    elif dim == 2:
        return surface / (2 * π)  # type: ignore
    elif dim == 3:
        return np.sqrt(surface / (4 * π))  # type: ignore
    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")


def make_surface_from_radius_compiled(dim: int) -> Callable[[TNumArr], TNumArr]:
    """Return a function calculating the surface area of a sphere.

    Args:
        dim (int): Dimension of the space

    Returns:
        function: A function that takes a radius and returns the surface area
    """
    import numba as nb

    if dim == 1:

        def _surface_from_radius(radius):
            if isinstance(radius, np.ndarray):
                return np.full(radius.shape, 2)
            else:
                return 2

        @overload(_surface_from_radius)
        def ol_surface_from_radius(radius):
            if isinstance(radius, nb.types.Array):
                return lambda radius: np.full(radius.shape, 2)
            else:
                return lambda radius: 2

        @jit
        def surface_from_radius(radius: TNumArr) -> TNumArr:
            return _surface_from_radius(radius)  # type: ignore

    elif dim == 2:

        @jit
        def surface_from_radius(radius: TNumArr) -> TNumArr:
            return 2 * π * radius  # type: ignore

    elif dim == 3:

        @jit
        def surface_from_radius(radius: TNumArr) -> TNumArr:
            return 4 * π * radius**2  # type: ignore

    else:
        raise NotImplementedError(
            f"Cannot calculate the surface area in {dim} dimensions"
        )
    return surface_from_radius  # type: ignore


def points_cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """Convert points from Cartesian to spherical coordinates.

    Args:
        points (:class:`~numpy.ndarray`): Points in Cartesian coordinates

    Returns:
        :class:`~numpy.ndarray`: Points (r, θ, φ) in spherical coordinates
    """
    points = np.atleast_1d(points)
    if points.shape[-1] != 3:
        raise DimensionError("Points must have 3 coordinates")

    ps_spherical = np.empty(points.shape)
    # calculate radius in [0, infinity]
    ps_spherical[..., 0] = np.linalg.norm(points, axis=-1)
    # calculate θ in [0, pi]
    ps_spherical[..., 1] = np.arccos(points[..., 2] / ps_spherical[..., 0])
    # calculate φ in [0, 2 * pi]
    ps_spherical[..., 2] = np.arctan2(points[..., 1], points[..., 0]) % (2 * π)
    return ps_spherical  # type: ignore


def points_spherical_to_cartesian(points: np.ndarray) -> np.ndarray:
    """Convert points from spherical to Cartesian coordinates.

    Args:
        points (:class:`~numpy.ndarray`):
            Points in spherical coordinates (r, θ, φ)

    Returns:
        :class:`~numpy.ndarray`: Points in Cartesian coordinates
    """
    points = np.atleast_1d(points)
    if points.shape[-1] != 3:
        raise DimensionError("Points must have 3 coordinates")

    sin_θ = np.sin(points[..., 1])
    ps_cartesian = np.empty(points.shape)
    ps_cartesian[..., 0] = points[..., 0] * np.cos(points[..., 2]) * sin_θ
    ps_cartesian[..., 1] = points[..., 0] * np.sin(points[..., 2]) * sin_θ
    ps_cartesian[..., 2] = points[..., 0] * np.cos(points[..., 1])
    return ps_cartesian  # type: ignore


@t.overload
def polar_coordinates(
    grid: GridBase,
    *,
    origin: np.ndarray | None = None,
    ret_angle: Literal[False] = False,
) -> np.ndarray: ...


@t.overload
def polar_coordinates(
    grid: GridBase, *, origin: np.ndarray | None = None, ret_angle: Literal[True]
) -> tuple[np.ndarray, ...]: ...


def polar_coordinates(
    grid: GridBase, *, origin: np.ndarray | None = None, ret_angle: bool = False
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return polar coordinates associated with grid points.

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid whose cell coordinates are used.
        origin (:class:`~numpy.ndarray`, optional):
            Cartesian coordinates of the origin at which polar coordinates are anchored.
        ret_angle (bool):
            Determines whether angles are returned alongside the distance. If `False`
            only the distance to the origin is returned for each support point of the
            grid. If `True`, the distance and angles are returned. For a 1d system
            system, the angle is defined as the sign of the difference between the
            point and the origin, so that angles can either be 1 or -1. For 2d
            systems and 3d systems, polar coordinates and spherical coordinates are
            used, respectively.

    Returns:
        :class:`~numpy.ndarray` or tuple of :class:`~numpy.ndarray`:
            Coordinates values in polar coordinates
    """
    if origin is None:
        origin = np.zeros(grid.dim)
    else:
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (grid.dim,):  # type: ignore
            raise DimensionError("Dimensions are not compatible")

    # calculate the difference vector between all cells and the origin
    origin_grid = grid.transform(origin, source="cartesian", target="grid")  # type: ignore
    diff = grid.difference_vector(origin_grid, grid.cell_coords)
    dist: np.ndarray = np.linalg.norm(diff, axis=-1)  # get distance

    # determine distance and optionally angles for these vectors
    if not ret_angle:
        return dist

    elif grid.dim == 1:
        return dist, np.sign(diff)[..., 0]

    elif grid.dim == 2:
        return dist, np.arctan2(diff[..., 1], diff[..., 0])

    elif grid.dim == 3:
        theta = np.arccos(diff[..., 2] / dist)
        phi = np.arctan2(diff[..., 1], diff[..., 0])
        return dist, theta, phi

    else:
        raise NotImplementedError(f"Cannot calculate angles for dimension {grid.dim}")


def spherical_index_k(degree: int, order: int = 0) -> int:
    """Returns the mode `k` from the degree `degree` and order `order`

    Args:
        degree (int):
            Degree :math:`l` of the spherical harmonics
        order (int):
            Order :math:`m` of the spherical harmonics

    Raises:
        ValueError: if `order < -degree` or `order > degree`

    Returns:
        int: a combined index k
    """
    if not -degree <= order <= degree:
        raise ValueError("order must lie between -degree and degree")
    return degree * (degree + 1) + order


def spherical_index_lm(k: int) -> tuple[int, int]:
    """Returns the degree `l` and the order `m` from the mode `k`

    Args:
        k (int):
            The combined index for the spherical harmonics

    Returns:
        tuple: The degree `l` and order `m` of the spherical harmonics
        associated with the combined index
    """
    degree = int(np.floor(np.sqrt(k)))
    return degree, k - degree * (degree + 1)


def spherical_index_count(l: int) -> int:
    """Return the number of modes for all indices <= l.

    The returned value is one less than the maximal mode `k` required.

    Args:
        l (int):
            Maximal degree of the spherical harmonics

    Returns:
        int: The number of modes
    """
    return 1 + 2 * l + l * l


def spherical_index_count_optimal(k_count: int) -> bool:
    """Checks whether the modes captures all orders for maximal degree.

    Args:
        k_count (int):
            The number of modes considered

    Returns:
        bool: indicates whether `k_count` is optimally chosen.
    """
    is_square = bool(int(np.sqrt(k_count) + 0.5) ** 2 == k_count)
    return is_square


def spherical_harmonic_symmetric(degree: int, θ: float) -> float:
    r"""Axisymmetric spherical harmonics with degree `degree`, so `m=0`.

    Args:
        degree (int):
            Degree of the spherical harmonics
        θ (float):
            Azimuthal angle at which function is evaluated (in :math:`[0, \pi]`)

    Returns:
        float: The value of the spherical harmonics
    """
    return np.real(sph_harm_y(degree, 0, θ, 0.0))  # type: ignore


def spherical_harmonic_real(degree: int, order: int, θ: float, φ: float) -> float:
    r"""Real spherical harmonics of degree l and order m.

    Args:
        degree (int):
            Degree :math:`l` of the spherical harmonics
        order (int):
            Order :math:`m` of the spherical harmonics
        θ (float):
            Azimuthal angle (in :math:`[0, \pi]`) at which function is evaluated.
        φ (float):
            Polar angle (in :math:`[0, 2\pi]`) at which function is evaluated.

    Returns:
        float: The value of the spherical harmonics
    """
    if order > 0:
        return (-1) ** order * np.sqrt(2) * np.real(sph_harm_y(degree, order, θ, φ))  # type: ignore

    elif order == 0:
        return np.real(sph_harm_y(degree, 0, θ, φ))  # type: ignore

    else:  # order < 0
        return np.sqrt(2) * (-1) ** order * np.imag(sph_harm_y(degree, -order, θ, φ))  # type: ignore


def spherical_harmonic_real_k(k: int, θ: float, φ: float) -> float:
    r"""Real spherical harmonics described by mode k.

    Args:
        k (int):
            Combined index determining the degree and order of the spherical harmonics
        θ (float):
            Azimuthal angle (in :math:`[0, \pi]`) at which function is evaluated.
        φ (float):
            Polar angle (in :math:`[0, 2\pi]`) at which function is evaluated.

    Returns:
        float: The value of the spherical harmonics
    """
    return spherical_harmonic_real(*spherical_index_lm(k), θ=θ, φ=φ)
