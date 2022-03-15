r"""
Module collecting functions for handling spherical geometry

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
   spherical_index_k
   spherical_index_lm
   spherical_index_count
   spherical_index_count_optimal
   spherical_harmonic_symmetric
   spherical_harmonic_real
   spherical_harmonic_real_k
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>    
"""

from typing import Callable, Tuple, TypeVar

import numpy as np
from scipy.special import sph_harm

from pde.grids.spherical import volume_from_radius
from pde.tools.numba import jit

π = float(np.pi)

TNumArr = TypeVar("TNumArr", float, np.ndarray)


def radius_from_volume(volume: TNumArr, dim: int) -> TNumArr:
    """Return the radius of a sphere with a given volume

    Args:
        volume (float or :class:`~numpy.ndarray`): Volume of the sphere
        dim (int): Dimension of the space

    Returns:
        float or :class:`~numpy.ndarray`: Radius of the sphere
    """
    if dim == 1:
        return volume / 2
    elif dim == 2:
        return np.sqrt(volume / π)  # type: ignore
    elif dim == 3:
        return (3 * volume / (4 * π)) ** (1 / 3)  # type: ignore
    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")


def make_radius_from_volume_compiled(dim: int) -> Callable[[TNumArr], TNumArr]:
    """Return a function calculating the radius of a sphere with a given volume

    Args:
        dim (int): Dimension of the space

    Returns:
        function: A function that takes a volume and returns the radius
    """
    if dim == 1:

        def radius_from_volume(volume):
            return volume / 2

    elif dim == 2:

        def radius_from_volume(volume):
            return np.sqrt(volume / π)

    elif dim == 3:

        def radius_from_volume(volume):
            return (3 * volume / (4 * π)) ** (1 / 3)

    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")
    return jit(radius_from_volume)  # type: ignore


def make_volume_from_radius_compiled(dim: int) -> Callable[[TNumArr], TNumArr]:
    """Return a function calculating the volume of a sphere with a given radius

    Args:
        dim (int): Dimension of the space

    Returns:
        function: A function that takes a radius and returns the volume
    """
    if dim == 1:

        def volume_from_radius(radius: TNumArr) -> TNumArr:
            return 2 * radius

    elif dim == 2:

        def volume_from_radius(radius: TNumArr) -> TNumArr:
            return π * radius**2

    elif dim == 3:

        def volume_from_radius(radius: TNumArr) -> TNumArr:
            return 4 * π / 3 * radius**3

    else:
        raise NotImplementedError(f"Cannot calculate the volume in {dim} dimensions")
    return jit(volume_from_radius)  # type: ignore


def surface_from_radius(radius: TNumArr, dim: int) -> TNumArr:
    """Return the surface area of a sphere with a given radius

    Args:
        radius (float or :class:`~numpy.ndarray`): Radius of the sphere
        dim (int): Dimension of the space

    Returns:
        float or :class:`~numpy.ndarray`: Surface area of the sphere
    """
    if dim == 1:
        if isinstance(radius, np.ndarray):
            return np.broadcast_to(2, radius.shape)  # type: ignore
        else:
            return 2
    elif dim == 2:
        return 2 * π * radius
    elif dim == 3:
        return 4 * π * radius**2
    else:
        raise NotImplementedError(
            f"Cannot calculate the surface area in {dim} dimensions"
        )


def radius_from_surface(surface: TNumArr, dim: int) -> TNumArr:
    """Return the radius of a sphere with a given surface area

    Args:
        surface (float or :class:`~numpy.ndarray`): Surface area of the sphere
        dim (int): Dimension of the space

    Returns:
        float or :class:`~numpy.ndarray`: Radius of the sphere
    """
    if dim == 1:
        raise RuntimeError("Cannot calculate radius of 1-d sphere from surface")
    elif dim == 2:
        return surface / (2 * π)
    elif dim == 3:
        return np.sqrt(surface / (4 * π))  # type: ignore
    else:
        raise NotImplementedError(f"Cannot calculate the radius in {dim} dimensions")


def make_surface_from_radius_compiled(dim: int) -> Callable[[TNumArr], TNumArr]:
    """Return a function calculating the surface area of a sphere

    Args:
        dim (int): Dimension of the space

    Returns:
        function: A function that takes a radius and returns the surface area
    """
    import numba as nb

    if dim == 1:

        if nb.config.DISABLE_JIT:
            # jitting is disabled => return generic python function
            def surface_from_radius(radius: TNumArr) -> TNumArr:
                if isinstance(radius, np.ndarray):
                    return np.full(radius.shape, 2)
                else:
                    return 2

        else:
            # jitting is enabled => return specific compiled functions
            @nb.generated_jit(nopython=True)
            def surface_from_radius(radius: TNumArr) -> TNumArr:
                if isinstance(radius, nb.types.Float):
                    return lambda radius: 2  # type: ignore
                else:
                    return lambda radius: np.full(radius.shape, 2)  # type: ignore

    elif dim == 2:

        @jit
        def surface_from_radius(radius: TNumArr) -> TNumArr:
            return 2 * π * radius

    elif dim == 3:

        @jit
        def surface_from_radius(radius: TNumArr) -> TNumArr:
            return 4 * π * radius**2

    else:
        raise NotImplementedError(
            f"Cannot calculate the surface area in {dim} dimensions"
        )
    return surface_from_radius


def points_cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """Convert points from Cartesian to spherical coordinates

    Args:
        points (:class:`~numpy.ndarray`): Points in Cartesian coordinates

    Returns:
        :class:`~numpy.ndarray`: Points (r, θ, φ) in spherical coordinates
    """
    points = np.atleast_1d(points)
    assert points.shape[-1] == 3, "Points must have 3 coordinates"

    ps_spherical = np.empty(points.shape)
    # calculate radius in [0, infinity]
    ps_spherical[..., 0] = np.linalg.norm(points, axis=-1)
    # calculate θ in [0, pi]
    ps_spherical[..., 1] = np.arccos(points[..., 2] / ps_spherical[..., 0])
    # calculate φ in [0, 2 * pi]
    ps_spherical[..., 2] = np.arctan2(points[..., 1], points[..., 0]) % (2 * π)
    return ps_spherical


def points_spherical_to_cartesian(points: np.ndarray) -> np.ndarray:
    """Convert points from spherical to Cartesian coordinates

    Args:
        points (:class:`~numpy.ndarray`):
            Points in spherical coordinates (r, θ, φ)

    Returns:
        :class:`~numpy.ndarray`: Points in Cartesian coordinates
    """
    points = np.atleast_1d(points)
    assert points.shape[-1] == 3, "Points must have 3 coordinates"

    sin_θ = np.sin(points[..., 1])
    ps_cartesian = np.empty(points.shape)
    ps_cartesian[..., 0] = points[..., 0] * np.cos(points[..., 2]) * sin_θ
    ps_cartesian[..., 1] = points[..., 0] * np.sin(points[..., 2]) * sin_θ
    ps_cartesian[..., 2] = points[..., 0] * np.cos(points[..., 1])
    return ps_cartesian


def spherical_index_k(degree: int, order: int = 0) -> int:
    """returns the mode `k` from the degree `degree` and order `order`

    Args:
        degree (int): Degree of the spherical harmonics
        order (int): Order of the spherical harmonics

    Raises:
        ValueError: if `order < -degree` or `order > degree`

    Returns:
        int: a combined index k
    """
    if not -degree <= order <= degree:
        raise ValueError("order must lie between -degree and degree")
    return degree * (degree + 1) + order


def spherical_index_lm(k: int) -> Tuple[int, int]:
    """returns the degree `l` and the order `m` from the mode `k`

    Args:
        k (int): The combined index for the spherical harmonics

    Returns:
        tuple: The degree `l` and order `m` of the spherical harmonics
        assoicated with the combined index
    """
    degree = int(np.floor(np.sqrt(k)))
    return degree, k - degree * (degree + 1)


def spherical_index_count(l: int) -> int:
    """return the number of modes for all indices <= l

    The returned value is one less than the maximal mode `k` required.

    Args:
        l (int): Maximal degree of the spherical harmonics

    Returns:
        int: The number of modes
    """
    return 1 + 2 * l + l * l


def spherical_index_count_optimal(k_count: int) -> bool:
    """checks whether the modes captures all orders for maximal degree

    Args:
        k_count (int): The number of modes considered
    """
    is_square = bool(int(np.sqrt(k_count) + 0.5) ** 2 == k_count)
    return is_square


def spherical_harmonic_symmetric(degree: int, θ: float) -> float:
    r"""axisymmetric spherical harmonics with degree `degree`, so `m=0`.

    Args:
        degree (int): Degree of the spherical harmonics
        θ (float): Azimuthal angle at which the spherical harmonics is
            evaluated (in :math:`[0, \pi]`)

    Returns:
        float: The value of the spherical harmonics
    """
    # note that the definition of `sph_harm` has a different convention for the
    # usage of the variables φ and θ and we thus have to swap the args
    return np.real(sph_harm(0.0, degree, 0.0, θ))  # type: ignore


def spherical_harmonic_real(degree: int, order: int, θ: float, φ: float) -> float:
    r"""real spherical harmonics of degree l and order m

    Args:
        degree (int): Degree :math:`l` of the spherical harmonics
        order (int): Order :math:`m` of the spherical harmonics
        θ (float): Azimuthal angle (in :math:`[0, \pi]`) at which the
            spherical harmonics is evaluated.
        φ (float): Polar angle (in :math:`[0, 2\pi]`) at which the spherical
            harmonics is evaluated.

    Returns:
        float: The value of the spherical harmonics
    """
    # note that the definition of `sph_harm` has a different convention for the
    # usage of the variables φ and θ and we thus have to swap the args
    # Moreover, the scipy functions expect first the order and then the degree
    if order > 0:
        term1 = sph_harm(order, degree, φ, θ)
        term2 = (-1) ** order * sph_harm(-order, degree, φ, θ)
        return np.real((term1 + term2) / np.sqrt(2))  # type: ignore

    elif order == 0:
        return np.real(sph_harm(0, degree, φ, θ))  # type: ignore

    else:  # order < 0
        term1 = sph_harm(-order, degree, φ, θ)
        term2 = (-1) ** order * sph_harm(order, degree, φ, θ)
        return np.real((term1 - term2) / (complex(0, np.sqrt(2))))  # type: ignore


def spherical_harmonic_real_k(k: int, θ: float, φ: float) -> float:
    r"""real spherical harmonics described by mode k

    Args:
        k (int): Combined index determining the degree and order of the
            spherical harmonics
        θ (float): Azimuthal angle (in :math:`[0, \pi]`) at which the
            spherical harmonics is evaluated.
        φ (float): Polar angle (in :math:`[0, 2\pi]`) at which the spherical
            harmonics is evaluated.

    Returns:
        float: The value of the spherical harmonics
    """
    return spherical_harmonic_real(*spherical_index_lm(k), θ=θ, φ=φ)
