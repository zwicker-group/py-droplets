"""
Classes representing (perturbed) droplets in various dimensions


.. autosummary::
   :nosignatures:

   SphericalDroplet
   DiffuseDroplet
   PerturbedDroplet2D
   PerturbedDroplet3D


Inheritance structure of the classes:

.. inheritance-diagram:: droplets.droplets
   :parts: 1

The details of the classes are explained below:

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, TypeVar

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy import integrate

from pde.fields import ScalarField
from pde.grids.base import GridBase
from pde.tools import spherical
from pde.tools.cuboid import Cuboid
from pde.tools.misc import preserve_scalars
from pde.tools.plotting import plot_on_axes

# work-around to satisfy type checking in python 3.6
if TYPE_CHECKING:
    # TYPE_CHECKING will always be False in production and this circular import
    # will thus be resolved.
    from .emulsions import EmulsionTimeCourse  # @UnusedImport


TDroplet = TypeVar("TDroplet", bound="DropletBase")


def get_dtype_field_size(dtype, field_name: str) -> int:
    """return the number of elements in a field of structured numpy array

    Args:
        dtype (list):
            The dtype of the numpy array
        field_name (str):
            The name of the field that needs to be checked
    """
    shape = dtype.fields[field_name][0].shape
    return np.prod(shape) if shape else 1  # type: ignore


def iterate_in_pairs(it, fill=0):
    """return consecutive pairs from an iterator

    For instance, `list(pair_iterator('ABCDE', fill='Z'))` returns
    `[('A', 'B'), ('C', 'D'), ('E', 'Z')]`

    Args:
        it (iterator): The iterator
        fill: The value returned for the second part of the last returned tuple
            if the length of the iterator is odd

    Returns:
        This is a generator function that yields pairs of items of the iterator
    """
    it = iter(it)
    while True:
        # obtain first item of tuple
        try:
            first = next(it)
        except StopIteration:
            break
        # obtain second item of tuple
        try:
            yield first, next(it)
        except StopIteration:
            yield first, fill
            break


class DropletBase:
    """represents a generic droplet

    The data associated with a droplet is stored in structured array.
    Consequently, the `dtype` of the array determines what information the
    droplet class stores.
    """

    _subclasses: Dict[str, "DropletBase"] = {}  # collect all inheriting classes

    __slots__ = ["data"]

    data: np.recarray  # all information about the droplet in a record array

    @classmethod
    def from_data(cls, data: np.recarray) -> "DropletBase":
        """create droplet class from a given data

        Args:
            data (:class:`numpy.recarray`):
                The data array used to initialize the droplet
        """
        # note that we do not call the __init__ method of the class, since we do
        # not need to invent the dtype and set all the attributes. We here
        # simply assume that the given data is sane
        obj = cls.__new__(cls)
        obj.data = data
        return obj  # type: ignore

    @classmethod
    def from_droplet(cls, droplet: "DropletBase", **kwargs) -> "DropletBase":
        r"""return a droplet with data taken from `droplet`

        Args:
            droplet (:class:`DropletBase`):
                Another droplet from which data is copied. This does not be the
                same type of droplet
            \**kwargs:
                Additional arguments that can overwrite data in `droplet` or
                set additional arguments for the current class
        """
        args = droplet._args
        args.update(kwargs)
        return cls(**args)  # type: ignore

    @classmethod
    @abstractmethod
    def get_dtype(cls, **kwargs):
        """determine the dtype representing this droplet class

        Returns:
            :class:`numpy.dtype`: the (structured) dtype associated with this class
        """
        pass

    def _init_data(self, **kwargs) -> None:
        """initializes the `data` attribute if it is not present

        Args:
            **kwargs:
                Arguments used to determine the dtype of the data array
        """
        if not hasattr(self, "data"):
            dtype = self.get_dtype(**kwargs)

            # We need to create an empty record with the correct data type. Note
            # that the conversion np.recarray(1)[0] turns the array into a
            # scalar type (instance of numpy.record) that contains the
            # structured data. This conversion is necessary for numba to operate
            # on the data.
            self.data = np.recarray(1, dtype=dtype)[0]

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return np.allclose(
            self._data_array, other._data_array, rtol=0, atol=0, equal_nan=True
        )

    def check_data(self):
        """ method that checks the validity and consistency of self.data """
        pass

    @property
    def _args(self):
        return {key: self.data[key] for key in self.data.dtype.names}

    def __str__(self):
        arg_list = [f"{key}={value}" for key, value in self._args.items()]
        return f"{self.__class__.__name__}({', '.join(arg_list)})"

    __repr__ = __str__

    @property
    def _data_array(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the data of the droplet in an unstructured array """
        return structured_to_unstructured(self.data)  # type: ignore

    def copy(self: TDroplet, **kwargs) -> TDroplet:
        r"""return a copy of the current droplet

        Args:
            \**kwargs: Additional arguments an be used to set data of the
                returned droplet.
        """
        if kwargs:
            return self.from_droplet(self, **kwargs)  # type: ignore
        else:
            return self.from_data(self.data.copy())  # type: ignore

    @property
    def data_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """tuple: lower and upper bounds on the parameters """
        num = len(self._data_array)
        return np.full(num, -np.inf), np.full(num, np.inf)


class SphericalDroplet(DropletBase):  # lgtm [py/missing-equals]
    """ Represents a single, spherical droplet """

    __slots__ = ["data"]

    def __init__(self, position: np.ndarray, radius: float):
        r"""
        Args:
            position (:class:`~numpy.ndarray`):
                Position of the droplet center
            radius (float):
                Radius of the droplet
        """
        self._init_data(position=position)

        self.position = position
        self.radius = radius
        self.check_data()

    def check_data(self):
        """ method that checks the validity and consistency of self.data """
        if self.radius < 0:
            raise ValueError("Radius must be positive")

    @classmethod
    def get_dtype(cls, **kwargs):
        """determine the dtype representing this droplet class

        Args:
            position (:class:`~numpy.ndarray`):
                The position vector of the droplet. This is used to determine the space
                dimension.

        Returns:
            :class:`numpy.dtype`: the (structured) dtype associated with this class
        """
        position = np.atleast_1d(kwargs.pop("position"))
        assert not kwargs  # no more keyword arguments
        dim = len(position)
        return [("position", float, (dim,)), ("radius", float)]

    @property
    def dim(self) -> int:
        """int: the spatial dimension this droplet is embedded in """
        return get_dtype_field_size(self.data.dtype, "position")

    @property
    def data_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """tuple: lower and upper bounds on the parameters """
        l, h = super().data_bounds
        l[self.dim] = 0  # radius must be non-negative
        return l, h

    @classmethod
    def from_volume(cls, position: np.ndarray, volume: float):
        """Construct a droplet from given volume instead of radius

        Args:
            position (:class:`~numpy.ndarray`): center of the droplet
            volume (float): volume of the droplet
            interface_width (float, optional): width of the interface
        """
        dim = len(np.array(position, np.double, ndmin=1))
        radius = spherical.radius_from_volume(volume, dim)
        return cls(position, radius)

    @property
    def position(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the position of the droplet """
        return self.data["position"]  # type: ignore

    @position.setter
    def position(self, value: np.ndarray):
        value = np.asanyarray(value)
        if len(value) != self.dim:
            raise ValueError(f"The dimension of the position must be {self.dim}")
        self.data["position"] = value

    @property
    def radius(self) -> float:
        """float: the radius of the droplet """
        return float(self.data["radius"])

    @radius.setter
    def radius(self, value: float):
        self.data["radius"] = value
        self.check_data()

    @property
    def volume(self) -> float:
        """float: volume of the droplet """
        return spherical.volume_from_radius(self.radius, self.dim)

    @volume.setter
    def volume(self, volume: float):
        """set the radius from a supplied volume """
        self.radius = spherical.radius_from_volume(volume, self.dim)

    @property
    def surface_area(self) -> float:
        """float: surface area of the droplet """
        return spherical.surface_from_radius(self.radius, self.dim)

    @property
    def bbox(self) -> Cuboid:
        """:class:`~pde.tools.cuboid.Cuboid`: bounding box of the droplet """
        return Cuboid.from_points(
            self.position - self.radius, self.position + self.radius
        )

    def overlaps(self, other: "SphericalDroplet", grid: GridBase = None) -> bool:
        """determine whether another droplet overlaps with this one

        Note that this function so far only compares the distances of the
        droplets to their radii, which does not respect perturbed droplets
        correctly.

        Args:
            other (:class:`~droplets.droplets.SphericalDroplet`):
                instance of the other droplet
            grid (:class:`~pde.grids.base.GridBase`):
                grid that determines how distances are measured, which is for instance
                important to respect periodic boundary conditions. If omitted, an
                Eucledian distance is assumed.

        Returns:
            bool: whether the droplets overlap or not
        """
        if grid is None:
            distance = np.linalg.norm(self.position - other.position)
        else:
            distance = grid.distance_real(self.position, other.position)
        return distance < self.radius + other.radius  # type: ignore

    @preserve_scalars
    def interface_position(self, *args) -> np.ndarray:
        r"""calculates the position of the interface of the droplet

        Args:
            *args (float or :class:`~numpy.ndarray`):
                The angles identifying the interface points. For 2d droplets, this is
                simply the angle in polar coordinates. For 3d droplets, both the
                azimuthal angle θ (in :math:`[0, \pi]`) and the polar angle φ (in
                :math:`[0, 2\pi]`) need to be specified.

        Returns:
            :class:`~numpy.ndarray`: An array with the coordinates of the interfacial
            points associated with each angle given by `φ`.

        Raises:
            ValueError: If the dimension of the space is not 2
        """
        if self.dim != len(args) + 1:
            raise ValueError(f"Interfacial position requires {self.dim - 1} angles")

        if self.dim == 2:
            # spherical droplet in two dimensions
            φ = args[0]
            pos = self.radius * np.transpose([np.cos(φ), np.sin(φ)])

        elif self.dim == 3:
            # spherical droplet in three dimensions
            θ, φ = args[0], args[1]
            r = np.full_like(θ, self.radius)
            pos = spherical.points_spherical_to_cartesian(np.c_[r, θ, φ])

        else:
            raise NotImplementedError(f"Cannot calculate {self.dim}d position")

        # shift the droplet center
        return self.position[None, :] + pos  # type: ignore

    @property
    def interface_curvature(self) -> float:
        """float: the mean curvature of the interface of the droplet"""
        return 1 / self.radius

    def _get_phase_field(self, grid: GridBase, dtype=np.double) -> np.ndarray:
        """Creates an image of the droplet on the `grid`

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field
            dtype (class:`numpy.dtype`):
                The numpy data type defining the type of the returned image.
                If `dtype == np.bool`, a binary representation is returned.

        Returns:
            :class:`~numpy.ndarray`: An array with data values representing the droplet
            phase field at support points of the `grid`.
        """
        if self.dim != grid.dim:
            raise ValueError(
                f"Droplet (dimension {self.dim}) incompatible with grid (dimension "
                f"{grid.dim})"
            )

        # calculate distances from droplet center
        dist = grid.polar_coordinates_real(self.position)
        return (dist < self.radius).astype(dtype)  # type: ignore

    def get_phase_field(
        self, grid: GridBase, *, vmin: float = 0, vmax: float = 1, label: str = None
    ) -> ScalarField:
        """Creates an image of the droplet on the `grid`

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field
            vmin (float):
                Minimal value the phase field will attain (far away from droplet)
            vmax (float):
                Maximal value the phase field will attain (inside the droplet)
            label (str):
                The label associated with the returned scalar field

        Returns:
            :class:`~pde.fields.ScalarField`: A scalar field
            representing the droplet
        """
        data = self._get_phase_field(grid)
        data = vmin + (vmax - vmin) * data  # scale data
        return ScalarField(grid, data=data, label=label)

    def get_triangulation(self, resolution: float = 1) -> Dict[str, Any]:
        """obtain a triangulated shape of the droplet surface

        Args:
            resolution (float):
                The length of a typical triangulation element. This affects the
                resolution of the triangulation.

        Returns:
            dict: A dictionary containing information about the triangulation. The exact
            details depend on the dimension of the problem.
        """
        if self.dim == 2:
            num = max(3, int(np.ceil(self.surface_area / resolution)))
            angles = np.linspace(0, 2 * np.pi, num + 1, endpoint=True)
            vertices = self.interface_position(angles)
            lines = np.c_[np.arange(num), np.arange(1, num + 1) % num]
            return {"vertices": vertices, "lines": lines}

        elif self.dim == 3:
            # estimate the number of triangles covering the surface
            try:
                surface_area = self.surface_area
            except (NotImplementedError, AttributeError):
                # estimate surface area from 3d spherical droplet
                surface_area = spherical.surface_from_radius(self.radius, dim=3)
            num_est = (4 * surface_area) / (np.sqrt(3) * resolution ** 2)
            tri = triangulated_spheres.get_triangulation(num_est)

            φ, θ = tri["angles"][:, 0], tri["angles"][:, 1]
            return {
                "vertices": self.interface_position(θ, φ),
                "triangles": tri["cells"],
            }

        else:
            raise NotImplementedError(f"Triangulation not implemented for {self.dim}d")

    def _get_mpl_patch(self, dim=None, **kwargs):
        """return the patch representing the droplet for plotting

        Args:
            dim (int, optional): The dimension in which the data is plotted. If omitted,
                the actual physical dimension is assumed
        """
        import matplotlib as mpl

        if dim is None:
            dim = self.dim

        if dim != 2:
            raise NotImplementedError("Plotting is only implemented in 2d")

        if self.dim == 1:
            position = (self.position[0], 0)
        else:
            position = self.position[:dim]

        return mpl.patches.Circle(position, self.radius, **kwargs)

    @plot_on_axes()
    def plot(self, ax=None, **kwargs):
        """Plot the droplet

        Args:
            {PLOT_ARGS}
            **kwargs:
                Additional keyword arguments are passed to the
                :class:`matplotlib.patches.Circle`, which creates the patch that
                represents the droplet
        """
        kwargs.setdefault("fill", False)
        ax.add_artist(self._get_mpl_patch(**kwargs))


class DiffuseDroplet(SphericalDroplet):
    """ Represents a single, spherical droplet with a diffuse interface """

    __slots__ = ["data"]

    def __init__(
        self, position: np.ndarray, radius: float, interface_width: float = None
    ):
        """
        Args:
            position (:class:`~numpy.ndarray`):
                Position of the droplet center
            radius (float):
                Radius of the droplet
            interface_width (float, optional):
                Width of the interface
        """
        self._init_data(position=position)
        super().__init__(position=position, radius=radius)
        self.interface_width = interface_width

    @property
    def data_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """tuple: lower and upper bounds on the parameters """
        l, h = super().data_bounds
        l[self.dim + 1] = 0  # interface width must be non-negative
        return l, h

    @classmethod
    def get_dtype(cls, **kwargs):
        """determine the dtype representing this droplet class

        Args:
            position (:class:`~numpy.ndarray`):
                The position vector of the droplet. This is used to determine the space
                dimension.

        Returns:
            :class:`numpy.dtype`: the (structured) dtype associated with this class
        """
        dtype = super().get_dtype(**kwargs)
        return dtype + [("interface_width", float)]

    @property
    def interface_width(self) -> Optional[float]:
        """float: the width of the interface of this droplet """
        if np.isnan(self.data["interface_width"]):
            return None
        else:
            return float(self.data["interface_width"])

    @interface_width.setter
    def interface_width(self, value: Optional[float]):
        if value is None:
            self.data["interface_width"] = np.nan
        elif value < 0:
            raise ValueError("Interface width must not be negative")
        else:
            self.data["interface_width"] = value
        self.check_data()

    def _get_phase_field(self, grid: GridBase, dtype=np.double) -> np.ndarray:
        """Creates an image of the droplet on the `grid`

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field
            dtype (:class:`numpy.dtype`):
                The numpy data type defining the type of the returned image.
                If `dtype == np.bool`, a binary representation is returned.

        Returns:
            :class:`~numpy.ndarray`: An array with data values representing the droplet
            phase field at support points of the `grid`.
        """
        if self.dim != grid.dim:
            raise ValueError(
                f"Droplet (dimension {self.dim}) incompatible with grid (dimension "
                f"{grid.dim})"
            )

        if self.interface_width is None:
            interface_width = grid.typical_discretization
        else:
            interface_width = self.interface_width

        # calculate distances from droplet center
        dist: np.ndarray = grid.polar_coordinates_real(self.position, ret_angle=False)  # type: ignore

        # make the image
        if interface_width == 0 or dtype == np.bool_:
            result = dist < self.radius
        else:
            result = 0.5 + 0.5 * np.tanh((self.radius - dist) / interface_width)

        return result.astype(dtype)  # type: ignore


class PerturbedDropletBase(DiffuseDroplet, metaclass=ABCMeta):
    """represents a single droplet with a perturbed shape.

    This acts as an abstract class for which member functions need to specified
    depending on dimensionality.
    """

    __slots__ = ["data"]

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        interface_width: float = None,
        amplitudes: np.ndarray = None,
    ):
        """
        Args:
            position (:class:`~numpy.ndarray`):
                Position of the droplet center
            radius (float):
                Radius of the droplet
            interface_width (float, optional):
                Width of the interface
            amplitudes (:class:`~numpy.ndarray`):
                The amplitudes of the perturbations
        """
        self._init_data(position=position, amplitudes=amplitudes)
        super().__init__(
            position=position, radius=radius, interface_width=interface_width
        )

        self.amplitudes = amplitudes  # type: ignore

        if len(self.position) != self.__class__.dim:
            raise ValueError(f"Space dimension must be {self.__class__.dim}")

    @classmethod
    def get_dtype(cls, **kwargs):
        """determine the dtype representing this droplet class

        Args:
            position (:class:`~numpy.ndarray`):
                The position vector of the droplet. This is used to determine the space
                dimension.
            amplitudes (:class:`~numpy.ndarray`):
                The perturbation amplitudes used to determine their size

        Returns:
            :class:`numpy.dtype`: the (structured) dtype associated with this class
        """
        # extract data
        amplitudes = kwargs.pop("amplitudes")
        if amplitudes is None:
            modes = 0
        else:
            modes = len(amplitudes)

        # create dtype
        dtype = super().get_dtype(**kwargs)
        return dtype + [("amplitudes", float, (modes,))]

    @property
    def data_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """tuple: lower and upper bounds on the parameters """
        l, h = super().data_bounds
        n = self.dim + 2
        # relative perturbation amplitudes must be between [-1, 1]
        l[n : n + self.modes] = -1
        h[n : n + self.modes] = 1
        return l, h

    @property
    def modes(self) -> int:
        """int: number of perturbation modes """
        shape = self.data.dtype.fields["amplitudes"][0].shape
        return int(shape[0]) if shape else 1

    @property
    def amplitudes(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the perturbation amplitudes """
        return np.atleast_1d(self.data["amplitudes"])

    @amplitudes.setter
    def amplitudes(self, value: np.ndarray = None):
        if value is None:
            assert self.modes == 0
            self.data["amplitudes"] = np.broadcast_to(0.0, (0,))
        else:
            self.data["amplitudes"] = np.broadcast_to(value, (self.modes,))
        self.check_data()

    @abstractmethod
    def interface_distance(self, *angles):
        pass

    @abstractmethod
    def interface_curvature(self, *angles):
        pass

    @property
    def volume(self) -> float:
        raise NotImplementedError

    @volume.setter
    def volume(self, volume: float):
        raise NotImplementedError

    @property
    def surface_area(self) -> float:
        raise NotImplementedError

    def _get_phase_field(self, grid: GridBase, dtype=np.double) -> np.ndarray:
        """Creates a normalized image of the droplet on the `grid`

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field

        Returns:
            :class:`~numpy.ndarray`: An array with data values representing the droplet
            phase field at support points of the `grid`.
        """
        if self.dim != grid.dim:
            raise ValueError(
                f"Droplet (dimension {self.dim}) incompatible with grid (dimension "
                f"{grid.dim})"
            )

        if self.interface_width is None:
            interface_width = grid.typical_discretization
        else:
            interface_width = self.interface_width

        # calculate grid distance from droplet center
        dist, *angles = grid.polar_coordinates_real(self.position, ret_angle=True)

        # calculate interface distance from droplet center
        interface = self.interface_distance(*angles)

        # make the image
        if interface_width == 0 or dtype == np.bool_:
            result = dist < interface
        else:
            result = 0.5 + 0.5 * np.tanh((interface - dist) / interface_width)

        return result.astype(dtype)  # type: ignore


class PerturbedDroplet2D(PerturbedDropletBase):
    r"""Represents a single droplet in two dimensions with a perturbed shape

    The shape is described using the distance :math:`R(\phi)` of the interface
    from the `position`, which is a function of the polar angle :math:`\phi`.
    This function is expressed as a truncated series of harmonics:

    .. math::
        R(\phi) = R_0 + R_0\sum_{n=1}^N \left[ \epsilon^{(1)}_n \sin(n \phi)
                                    + \epsilon^{(2)}_n \cos(n \phi) \right]

    where :math:`N` is the number of perturbation modes considered, which is
    given by half the length of the `amplitudes` array. Consequently, amplitudes
    should always be an even number, to consider both `sin` and `cos` terms.
    """

    dim = 2

    __slots__ = ["data"]

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        interface_width: float = None,
        amplitudes: np.ndarray = None,
    ):
        r"""
        Args:
            position (:class:`~numpy.ndarray`):
                Position of the droplet center
            radius (float):
                Radius of the droplet
            interface_width (float, optional):
                Width of the interface
            amplitudes (:class:`~numpy.ndarray`):
                (dimensionless) perturbation amplitudes
                :math:`\{\epsilon^{(1)}_1, \epsilon^{(2)}_1, \epsilon^{(1)}_2,
                \epsilon^{(2)}_2, \epsilon^{(1)}_3, \epsilon^{(2)}_3, \dots \}`.
                The length of the array needs to be even to capture
                perturbations of the highest mode consistently.
        """
        super().__init__(position, radius, interface_width, amplitudes)
        if len(self.amplitudes) % 2 != 0:
            logger = logging.getLogger(self.__class__.__name__)
            logger.warning(
                "`amplitudes` should be of even length to capture all perturbations of "
                "the highest mode."
            )

    @preserve_scalars
    def interface_distance(self, φ):
        """calculates the distance of the droplet interface to the origin

        Args:
            φ (float or array): The angle in the polar coordinate system that
                is used to describe the interface

        Returns:
            An array with the distances of the interfacial points associated
            with each angle given by `φ`.
        """
        dist = np.ones(φ.shape, dtype=np.double)
        for n, (a, b) in enumerate(iterate_in_pairs(self.amplitudes), 1):  # no 0th mode
            if a != 0:
                dist += a * np.sin(n * φ)
            if b != 0:
                dist += b * np.cos(n * φ)
        return self.radius * dist

    @preserve_scalars
    def interface_position(self, φ):
        """calculates the position of the interface of the droplet

        Args:
            φ (float or array): The angle in the polar coordinate system that
                is used to describe the interface

        Returns:
            An array with the coordinates of the interfacial points associated
            with each angle given by `φ`.
        """
        dist = self.interface_distance(φ)
        pos = dist[:, None] * np.transpose([np.sin(φ), np.cos(φ)])
        return self.position[None, :] + pos

    @preserve_scalars
    def interface_curvature(self, φ):
        r"""calculates the mean curvature of the interface of the droplet

        For simplicity, the effect of the perturbations are only included to
        linear order in the perturbation amplitudes :math:`\epsilon^{(1/2)}_n`.

        Args:
            φ (float or array): The angle in the polar coordinate system that
                is used to describe the interface

        Returns:
            An array with the curvature at the interfacial points associated
            with each angle given by `φ`.
        """
        curv_radius = np.ones(φ.shape, dtype=np.double)
        for n, (a, b) in enumerate(iterate_in_pairs(self.amplitudes), 1):  # no 0th mode
            factor = n * n - 1
            if a != 0:
                curv_radius -= a * factor * np.sin(n * φ)
            if b != 0:
                curv_radius -= b * factor * np.cos(n * φ)
        return 1 / (self.radius * curv_radius)

    @property
    def volume(self) -> float:
        """ float: volume of the droplet """
        term = 1 + np.sum(self.amplitudes ** 2) / 2
        return np.pi * self.radius ** 2 * term  # type: ignore

    @volume.setter
    def volume(self, volume: float):
        """ set volume keeping relative perturbations """
        term = 1 + np.sum(self.amplitudes ** 2) / 2
        self.radius = np.sqrt(volume / (np.pi * term))

    @property
    def surface_area(self) -> float:
        """ float: surface area of the droplet """
        # discretize surface for simple approximation to integral
        φs, dφ = np.linspace(0, 2 * np.pi, 256, endpoint=False, retstep=True)

        dist = np.ones(φs.shape, dtype=np.double)
        dist_dφ = np.zeros(φs.shape, dtype=np.double)
        for n, (a, b) in enumerate(iterate_in_pairs(self.amplitudes), 1):  # no 0th mode
            if a != 0:
                dist += a * np.sin(n * φs)
                dist_dφ += a * n * np.cos(n * φs)
            if b != 0:
                dist += b * np.cos(n * φs)
                dist_dφ -= b * n * np.sin(n * φs)

        dx = dist_dφ * np.cos(φs) - dist * np.sin(φs)
        dy = dist_dφ * np.sin(φs) + dist * np.cos(φs)
        line_element = np.hypot(dx, dy)

        return self.radius * line_element.sum() * dφ  # type: ignore

    @property
    def surface_area_approx(self) -> float:
        """ float: surface area of the droplet (quadratic in amplitudes) """
        length = 4
        for n, (a, b) in enumerate(iterate_in_pairs(self.amplitudes), 1):  # no 0th mode
            length += n ** 2 * (a ** 2 + b ** 2)
        return np.pi * self.radius * length / 2

    def _get_mpl_patch(self, dim=2, **kwargs):
        """return the patch representing the droplet for plotting

        Args:
            dim (int, optional): The dimension in which the data is plotted. If omitted,
                the actual physical dimension is assumed
        """
        import matplotlib as mpl

        if dim is None:
            dim = self.dim

        if dim != 2:
            raise NotImplementedError("Plotting is only implemented in 2d")

        φ = np.linspace(0, 2 * np.pi, endpoint=False)
        xy = self.interface_position(φ)
        return mpl.patches.Polygon(xy, closed=True, **kwargs)


class PerturbedDroplet3D(PerturbedDropletBase):
    r"""Represents a single droplet in two dimensions with a perturbed shape

    The shape is described using the distance :math:`R(\theta, \phi)` of the
    interface from the origin as a function of the azimuthal angle
    :math:`\theta` and the polar angle :math:`\phi`. This function is developed
    as a truncated series of spherical harmonics :math:`Y_{l,m}(\theta, \phi)`:

    .. math::
        R(\theta, \phi) = R_0 \left[1 + \sum_{l=1}^{N_l}\sum_{m=-l}^l
                                \epsilon_{l,m} Y_{l,m}(\theta, \phi) \right]

    where :math:`N_l` is the number of perturbation modes considered, which is
    deduced from the length of the `amplitudes` array.
    """

    dim = 3

    __slots__ = ["data"]

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        interface_width: float = None,
        amplitudes: np.ndarray = None,
    ):
        r"""
        Args:
            position (:class:`~numpy.ndarray`):
                Position of the droplet center
            radius (float):
                Radius of the droplet
            interface_width (float, optional):
                Width of the interface
            amplitudes (:class:`~numpy.ndarray`):
                Perturbation amplitudes :math:`\epsilon_{l,m}`. Note that the zero-th
                mode, which would only change the radius, is skipped. Consequently, the
                length of the array needs to be 0, 3, 8, 15, 24, ... to capture
                perturbations of the highest mode consistently.
        """
        super().__init__(position, radius, interface_width, amplitudes)
        num_modes = len(self.amplitudes) + 1
        if not spherical.spherical_index_count_optimal(num_modes):
            logger = logging.getLogger(self.__class__.__name__)
            l, _ = spherical.spherical_index_lm(num_modes)
            opt_modes = spherical.spherical_index_count(l) - 1
            logger.warning(
                "The length of `amplitudes` should be such that all orders are "
                f"captured for the perturbations with the highest degree ({l}). "
                f"Consider increasing the size of the array to {opt_modes}."
            )

    @preserve_scalars
    def interface_distance(self, θ, φ):
        r"""calculates the distance of the droplet interface to the origin

        Args:
            θ (float or array): Azimuthal angle (in :math:`[0, \pi]`)
            φ (float or array): Polar angle (in :math:`[0, 2\pi]`)

        Returns:
            An array with the distances of the interfacial points associated
            with the angles.
        """
        assert θ.shape == φ.shape
        dist = np.ones(φ.shape, dtype=np.double)
        for k, a in enumerate(self.amplitudes, 1):  # skip zero-th mode!
            if a != 0:
                dist += a * spherical.spherical_harmonic_real_k(k, θ, φ)
        return self.radius * dist

    @preserve_scalars
    def interface_position(self, θ, φ):
        r"""calculates the position of the interface of the droplet

        Args:
            θ (float or array): Azimuthal angle (in :math:`[0, \pi]`)
            φ (float or array): Polar angle (in :math:`[0, 2\pi]`)

        Returns:
            An array with the coordinates of the interfacial points associated
            with the angles.
        """
        dist = self.interface_distance(θ, φ)
        unit_vector = [np.sin(θ) * np.cos(φ), np.sin(θ) * np.sin(φ), np.cos(θ)]
        pos = dist[:, None] * np.transpose(unit_vector)
        return self.position[None, :] + pos

    @preserve_scalars
    def interface_curvature(self, θ, φ):
        r"""calculates the mean curvature of the interface of the droplet

        For simplicity, the effect of the perturbations are only included to
        linear order in the perturbation amplitudes :math:`\epsilon_{l,m}`.

        Args:
            θ (float or array): Azimuthal angle (in :math:`[0, \pi]`)
            φ (float or array): Polar angle (in :math:`[0, 2\pi]`)

        Returns:
            An array with the curvature at the interfacial points associated
            with the angles
        """
        Yk = spherical.spherical_harmonic_real_k
        correction = 0
        for k, a in enumerate(self.amplitudes, 1):  # skip zero-th mode!
            if a != 0:
                l, _ = spherical.spherical_index_lm(k)
                hk = (l ** 2 + l - 2) / 2
                correction = a * hk * Yk(k, θ, φ)
        return 1 / self.radius + correction / self.radius ** 2

    @property
    def volume(self) -> float:
        """ float: volume of the droplet (determined numerically) """

        def integrand(θ, φ):
            """ helper function calculating the integrand """
            r = self.interface_distance(θ, φ)
            return r ** 3 * np.sin(θ) / 3

        volume = integrate.dblquad(
            integrand, 0, 2 * np.pi, lambda _: 0, lambda _: np.pi
        )[0]
        return volume  # type: ignore

    @volume.setter
    def volume(self, volume: float):
        """ set volume keeping relative perturbations """
        raise NotImplementedError("Cannot set volume")

    @property
    def volume_approx(self) -> float:
        """ float: approximate volume to linear order in the perturbation """
        volume = spherical.volume_from_radius(self.radius, 3)
        if len(self.amplitudes) > 0:
            volume += self.amplitudes[0] * 2 * np.sqrt(np.pi) * self.radius ** 2
        return volume

    def _get_mpl_patch(self, dim=None, **kwargs):
        """return the patch representing the droplet for plotting

        Args:
            dim (int, optional): The dimension in which the data is plotted. If omitted,
                the actual physical dimension is assumed
        """
        raise NotImplementedError("Plotting PerturbedDroplet3D is not implemented")


def droplet_from_data(droplet_class: str, data) -> DropletBase:
    """create a droplet instance of the given class using some data

    Args:
        droplet_class (str): The name of the class that is used to create the
            droplet instance
        data (numpy.ndarray): A numpy array that defines the droplet properties
    """
    cls = DropletBase._subclasses[droplet_class]
    return cls(**{key: data[key] for key in data.dtype.names})  # type: ignore


class _TriangulatedSpheres:
    """helper class for handling stored data about triangulated spheres """

    def __init__(self):
        self.path = Path(__file__).resolve().parent / "resources" / "spheres_3d.hdf5"
        self.num_list = np.zeros((0,))
        self.data: Optional[Dict[int, Dict[str, Any]]] = None

    def _load(self):
        """load the stored resource """
        import h5py

        logger = logging.getLogger(__name__)
        logger.info("Open resource `%s`", self.path)
        with h5py.File(self.path, "r") as f:
            self.num_list = np.array(f.attrs["num_list"])
            self.data = {}
            for num in self.num_list:
                group = f[str(num)]
                tri = {
                    "points": np.array(group["points"]),
                    "angles": np.array(group["angles"]),
                    "cells": np.array(group["cells"]),
                }
                self.data[num] = tri

    def get_triangulation(self, num_est: int = 1) -> Dict[str, Any]:
        """get a triangulation of a sphere

        Args:
            num_est (int): The rough number of vertices in the triangulation

        Returns:
            dict: A dictionary with information about the triangulation
        """
        if self.data is None:
            self._load()

        index = np.argmin((self.num_list - num_est) ** 2)
        return self.data[self.num_list[index]]  # type: ignore


triangulated_spheres = _TriangulatedSpheres()


__all__ = [
    "SphericalDroplet",
    "DiffuseDroplet",
    "PerturbedDroplet2D",
    "PerturbedDroplet3D",
]
