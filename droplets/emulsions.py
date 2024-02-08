"""
Classes describing collections of droplets, i.e. emulsions, and their temporal dynamics.

.. autosummary::
   :nosignatures:

   Emulsion
   EmulsionTimeCourse

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
import json
import logging
import math
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

import numpy as np

from pde.fields import ScalarField
from pde.grids.base import GridBase
from pde.grids.cartesian import CartesianGrid
from pde.storage.base import StorageBase
from pde.tools.cuboid import Cuboid
from pde.tools.docstrings import fill_in_docstring
from pde.tools.output import display_progress
from pde.tools.plotting import PlotReference, plot_on_axes
from pde.trackers.base import InfoDict, InterruptData

from .droplets import SphericalDroplet, droplet_from_data

if TYPE_CHECKING:
    from .trackers import DropletTracker  # @UnusedImport


class Emulsion(list):
    """class representing a collection of droplets in a common system"""

    _show_projection_warning: bool = True
    """bool: Flag determining whether a warning is shown when high-dimensional
    emulsions are plotted"""

    def __init__(
        self,
        droplets: Iterable[SphericalDroplet] | None = None,
        *,
        copy: bool = True,
        dtype: np.typing.DTypeLike | np.ndarray | SphericalDroplet = None,
        force_consistency: bool = False,
        grid: GridBase | None = None,
    ):
        """
        Args:
            droplets:
                A list or generator of instances of
                :class:`~droplets.droplets.SphericalDroplet`.
            copy (bool, optional):
                Whether to make a copy of the droplet or not
            dtype (:class:`~numpy.tpying.DTypeLike`):
                The dtype describing what droplets are stored in the emulsion. Providing
                this is usually only necessary for creating empty emulsions. Instead of
                a dtype, an array or an example droplet can also be supplied.
            force_consistency (bool, optional):
                Whether to ensure that all droplets are of the same type, i.e., their
                data is described by the same dtype and matches `dtype` if given.
        """
        super().__init__()

        if grid is not None:
            # deprecated on 2023-08-29
            warnings.warn("`grid` argument is deprecated", DeprecationWarning)

        # store general information about droplets using a single dtype
        if isinstance(dtype, SphericalDroplet):
            dtype = dtype.data.dtype  # extract dtype from actual droplet
        elif isinstance(dtype, np.ndarray):
            dtype = dtype.dtype  # extract dtype from numpy array
        elif dtype is not None:
            dtype = np.dtype(dtype)  # assume a proper dtype is given
        assert dtype is None or isinstance(dtype, np.dtype)
        if isinstance(dtype, np.record):
            self.dtype: np.typing.DTypeLike = dtype.dtype  # strip record part
        else:
            self.dtype = dtype

        # add the actual droplets that are specified
        if droplets is not None:
            self.extend(droplets, copy=copy, force_consistency=force_consistency)

    @classmethod
    def empty(cls, droplet: SphericalDroplet) -> Emulsion:
        """create empty emulsion with particular droplet type

        Args:
            droplet (:class:`~droplets.droplets.SphericalDroplet`):
                An example for a droplet, which defines the type of

        Returns:
            :class:`Emulsion`: The empty emulsion
        """
        return cls([], dtype=droplet, copy=False)

    @classmethod
    def from_random(
        cls,
        num: int,
        grid_or_bounds: GridBase | Sequence[tuple[float, float]],
        radius: float | tuple[float, float],
        *,
        remove_overlapping: bool = True,
        droplet_class: type[SphericalDroplet] = SphericalDroplet,
        rng: np.random.Generator | None = None,
    ) -> Emulsion:
        """
        Create an emulsion with random droplets

        Args:
            num (int):
                The (maximal) number of droplets to generate
            grid_or_bounds (:class:`~pde.grids.base.GridBase` or list of float tuples):
                Determines the space in which droplets are placed. This is either a
                :class:`~pde.grids.base.GridBase` describing the geometry or a sequence
                of tuples with lower and upper bounds for each axes, so the length of
                the sequence determines the space dimension.
            radius (float or tuple of float):
                Radius of the droplets that are created. If two numbers are given, they
                specify the bounds of a uniform distribution from which the radius of
                each individual droplet is chosen.
            remove_overlapping (bool):
                Flag determining whether overlapping droplets are removed. If enabled,
                the resulting element might contain less thatn `num` droplets.
            droplet_class (:class:`~droplets.droplets.SphericalDroplet`):
                The class that is used to create droplets.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)

        Returns:
            :class:`Emulsion`: The emulsion containing the random droplets
        """
        if rng is None:
            rng = np.random.default_rng()

        # determine how to get random droplet positions
        if isinstance(grid_or_bounds, GridBase):

            def get_position():
                return grid_or_bounds.get_random_point(rng=rng)

        else:
            bnds = np.atleast_2d(grid_or_bounds)
            assert bnds.ndim == 2 and bnds.shape[0] > 0 and bnds.shape[1] == 2

            def get_position():
                return rng.uniform(bnds[:, 0], bnds[:, 1])

        # extract information about radius
        try:
            r0, r1 = radius  # type: ignore
        except TypeError:
            r0 = r1 = float(radius)  # type: ignore

        # create the emulsion from a list of droplets
        drops = [droplet_class(get_position(), rng.uniform(r0, r1)) for _ in range(num)]
        emulsion = cls(drops)

        if remove_overlapping:
            emulsion.remove_overlapping()

        return emulsion

    @property
    def dim(self) -> int | None:
        """int: dimensionality of space in which droplets are defined"""
        if self.dtype:
            return self.dtype["position"].shape[0]  # type: ignore
        else:
            return None

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __add__(self, rhs):
        # return sum of Emulsions as an emulsion
        return Emulsion(list.__add__(self, rhs))

    @overload  # type: ignore
    def __getitem__(self, key: int) -> SphericalDroplet: ...

    @overload
    def __getitem__(self, key: slice) -> Emulsion: ...

    def __getitem__(self, key):
        # return result from extended slicing as Emulsion
        result = list.__getitem__(self, key)
        if isinstance(result, SphericalDroplet):
            return result
        else:
            return Emulsion(result)

    def copy(self, min_radius: float = -1) -> Emulsion:
        """return a copy of this emulsion

        Args:
            min_radius (float):
                The minimal radius of the droplets that are retained. Droplets with
                exactly min_radius are removed, so `min_radius == 0` can be used to
                filter vanished droplets.
        """
        droplets: list[SphericalDroplet] = [
            droplet.copy() for droplet in self if droplet.radius > min_radius
        ]
        return self.__class__(droplets, copy=False)

    def extend(
        self,
        droplets: Iterable[SphericalDroplet],
        *,
        copy: bool = True,
        force_consistency: bool = False,
    ) -> None:
        """add many droplets to the emulsion

        Args:
            droplet (list of :class:`droplets.dropelts.SphericalDroplet`):
                List of droplets to add to the emulsion
            copy (bool, optional):
                Whether to make a copy of the droplet or not
            force_consistency (bool, optional):
                Whether to ensure that all droplets are of the same type
        """
        for droplet in droplets:
            self.append(droplet, copy=copy, force_consistency=force_consistency)

    def append(
        self,
        droplet: SphericalDroplet,
        *,
        copy: bool = True,
        force_consistency: bool = False,
    ) -> None:
        """add a droplet to the emulsion

        Args:
            droplet (:class:`droplets.dropelts.SphericalDroplet`):
                Droplet to add to the emulsion
            copy (bool, optional):
                Whether to make a copy of the droplet or not
            force_consistency (bool, optional):
                Whether to ensure that all droplets are of the same type
        """
        # during some multiprocessing examples, Emulsions might apparently not have
        # a proper `dtype` define. This might be because they use some copying or
        # __getstate__ methods of the underlying list class
        if not hasattr(self, "dtype") or self.dtype is None:
            self.dtype = droplet.data.dtype
        elif force_consistency and self.dtype != droplet.data.dtype:
            raise ValueError(
                f"Expected type {self.dtype}, but got {droplet.data.dtype}"
            )
        if copy:
            droplet = droplet.copy()
        super().append(droplet)

    @property
    def data(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: an array containing the data of the full emulsion

        Warning:
            This requires all droplets to be of the same class. The returned array is
            a copy of all the data and writing to it will thus not change the underlying
            data.
        """
        if len(self) == 0:
            # deal with empty emulsions
            if self.dtype:
                return np.empty(0, dtype=self.dtype)
            else:
                raise RuntimeError(
                    "Cannot create data array since the emulsion is empty and an "
                    "explicit dtype has not been specified."
                )

        else:
            # emulsion contains at least one droplet
            classes = {d.__class__ for d in self}
            if len(classes) > 1:
                raise TypeError(
                    "Emulsion data cannot be stored contiguously if it contains a "
                    f"multiple of droplet classes: "
                    + ", ".join(c.__name__ for c in classes)
                )
            result = np.array([d.data for d in self])
            if result.dtype != self.dtype:
                logger = logging.getLogger(self.__class__.__name__)
                logger.warning("Emulsion had inconsistent dtypes")
            return result

    def get_linked_data(self) -> np.ndarray:
        """link the data of all droplets in a single array

        Returns:
            :class:`~numpy.ndarray`: The array containing all droplet data. If entries in
                this array are modified, it will be reflected in the droplets.
        """
        data = self.data  # create an array with all the droplet data
        # link back to droplets
        for i, d in enumerate(self):
            d.data = data[i]
        return data

    @classmethod
    def _from_hdf_dataset(cls, dataset) -> Emulsion:
        """construct an emulsion by reading data from an hdf5 dataset

        Args:
            dataset:
                An HDF5 dataset from which the data of the emulsion is read

        Returns:
            :class:`Emulsion`: The emulsion read from the file
        """
        # there are values, so the emulsion is not empty
        droplet_class = dataset.attrs["droplet_class"]
        if droplet_class == "None":
            droplets: list[SphericalDroplet] = []
        else:
            droplets = [
                droplet_from_data(droplet_class, data)  # type: ignore
                for data in dataset
            ]
        return cls(droplets, copy=False)

    @classmethod
    def from_file(cls, path: str) -> Emulsion:
        """create emulsion by reading file

        Args:
            path (str):
                The path from which the data is read. This function assumes that the
                data was written as an HDF5 file using :meth:`to_file`.

        Returns:
            :class:`Emulsion`: The emulsion read from the file
        """
        import h5py

        with h5py.File(path, "r") as fp:
            if len(fp) != 1:
                raise RuntimeError(f"Multiple emulsions found in file `{path}`")

            # read the actual droplet data
            dataset = fp[list(fp.keys())[0]]  # retrieve the only dataset
            obj = cls._from_hdf_dataset(dataset)

        return obj

    def _write_hdf_dataset(self, hdf_path, key: str = "emulsion"):
        """write data to a given hdf5 path `hdf_path`

        Args:
            hdf_path:
                Location in an opened HDF file where the emulsion will be written
            key (str):
                Name of the data entry that will be written

        Returns:
            HDF element that stores the emulsion
        """
        if self:
            # emulsion contains at least one droplet
            dataset = hdf_path.create_dataset(key, data=self.data)
            # self.data ensures that there is only one droplet class
            dataset.attrs["droplet_class"] = self[0].__class__.__name__

        else:
            # create empty dataset to indicate empty emulsion
            dataset = hdf_path.create_dataset(key, shape=tuple())
            dataset.attrs["droplet_class"] = "None"

        return dataset

    def to_file(self, path: str) -> None:
        """store data in hdf5 file

        The data can be read using the classmethod :meth:`Emulsion.from_file`.

        Args:
            path (str):
                The path to which the data is written as an HDF5 file.
        """
        import h5py

        with h5py.File(path, "w") as fp:
            # write actual droplet data
            self._write_hdf_dataset(fp)

    @property
    def interface_width(self) -> float | None:
        """float: the average interface width across all droplets

        This averages the interface widths of the individual droplets weighted by their
        surface area, i.e., the amount of interface.
        """
        width, area = 0.0, 0.0
        for droplet in self:
            try:
                interface_width = droplet.interface_width
            except AttributeError:
                pass
            else:
                if interface_width is not None:
                    a = droplet.surface_area
                    width += interface_width * a
                    area += a

        if area == 0:
            return None
        else:
            return width / area

    @property
    def bbox(self) -> Cuboid:
        """:class:`Cuboid`: bounding box of the emulsion"""
        if len(self) == 0:
            raise RuntimeError("Bounding box of empty emulsion is undefined")
        return sum((droplet.bbox for droplet in self[1:]), self[0].bbox)

    def get_phasefield(self, grid: GridBase, label: str | None = None) -> ScalarField:
        """create a phase field representing a list of droplets

        Args:
            grid (:class:`pde.grids.base.GridBase`):
                The grid on which the phase field is created. If omitted, the grid
                associated with the emulsion is used.
            label (str, optional):
                Optional label for the returned scalar field

        Returns:
            :class:`~pde.fields.scalar.ScalarField`: the actual phase field
        """
        if len(self) == 0:
            return ScalarField(grid)

        else:
            result: ScalarField = self[0].get_phase_field(grid, label=label)
            for d in self[1:]:
                result += d.get_phase_field(grid)
            np.clip(result.data, 0, 1, out=result.data)
            return result

    def remove_small(self, min_radius: float = -np.inf) -> None:
        """remove droplets that are very small

        The emulsions is modified in-place.

        Args:
            min_radius (float):
                The minimal radius of the droplets that are retained. Droplets with
                exactly min_radius are removed, so `min_radius == 0` can be used to
                filter vanished droplets. The default value does not remove any droplets
        """
        for i in reversed(range(len(self))):
            if self[i].radius <= min_radius:
                self.pop(i)

    def get_pairwise_distances(
        self, subtract_radius: bool = False, grid: GridBase | None = None
    ) -> np.ndarray:
        """return the pairwise distance between droplets

        Args:
            subtract_radius (bool):
                Determines whether to subtract the radius from the distance, i.e.,
                whether to return the distance between the surfaces instead of the
                positions
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the droplets are defined, which is necessary if
                periodic boundary conditions should be respected for measuring distances

        Returns:
            :class:`~numpy.ndarray`: a matrix with the distances between all droplets
        """
        if grid is None:

            def get_distance(p1, p2):
                """helper function calculating the distance between points"""
                return np.linalg.norm(p1 - p2)

        else:
            get_distance = functools.partial(grid.distance, coords="cartesian")

        # calculate pairwise distance and return it in requested form
        num = len(self)
        dists = np.zeros((num, num))
        # iterate over all droplet pairs
        for i in range(num):
            for j in range(i + 1, num):
                d1, d2 = self[i], self[j]
                dist = get_distance(d1.position, d2.position)
                if subtract_radius:
                    dist -= d1.radius + d2.radius
                dists[i, j] = dists[j, i] = dist

        return dists

    def get_neighbor_distances(self, subtract_radius: bool = False) -> np.ndarray:
        """calculates the distance of each droplet to its nearest neighbor

        Warning:
            This function does not take periodic boundary conditions into account.

        Args:
            subtract_radius (bool):
                Determines whether to subtract the radius from the distance, i.e.,
                whether to return the distance between the surfaces instead of the
                positions

        Returns:
            :class:`~numpy.ndarray`: a vector with a distance for each droplet
        """
        # handle simple cases
        if len(self) == 0:
            return np.zeros((0,))
        elif len(self) == 1:
            return np.full(1, np.nan)

        try:
            from scipy.spatial import cKDTree as KDTree
        except ImportError:
            from scipy.spatial import KDTree

        # build tree to query the nearest neighbors
        assert self.data is not None
        positions = self.data["position"]

        # we could support periodic boundary conditions using `freud.locality.AABBQuery`
        tree = KDTree(positions)
        dist, index = tree.query(positions, 2)

        if subtract_radius:
            return dist[:, 1] - self.data["radius"][index].sum(axis=1)  # type: ignore
        else:
            return dist[:, 1]  # type: ignore

    def remove_overlapping(
        self, min_distance: float = 0, grid: GridBase | None = None
    ) -> None:
        """remove all droplets that are overlapping

        If a pair of overlapping droplets was found, the smaller one of these is removed
        from the current emulsion. This method modifies the emulsion in place and thus
        does not return anything.

        Args:
            min_distance (float):
                The minimal distance droplets need to be apart. The default value of `0`
                corresponds to just remove overlapping droplets. Larger values ensure
                that droplets keep a distance, while negative values allow for some
                overlap.
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the droplets are defined, which is necessary if
                periodic boundary conditions should be respected for measuring distances
        """
        # filter duplicates until there are none left
        dists = self.get_pairwise_distances(subtract_radius=True, grid=grid)
        np.fill_diagonal(dists, np.inf)

        while len(dists) > 1:
            # find minimal distance
            x, y = np.unravel_index(np.argmin(dists), dists.shape)
            if dists[x, y] < min_distance:
                # droplets overlap -> remove the smaller one
                if self[x].radius > self[y].radius:  # type: ignore
                    self.pop(y)
                    dists = np.delete(np.delete(dists, y, 0), y, 1)
                else:
                    self.pop(x)
                    dists = np.delete(np.delete(dists, x, 0), x, 1)
            else:
                break

    @property
    def total_droplet_volume(self) -> float:
        """float: the total volume of all droplets"""
        return sum(droplet.volume for droplet in self)  # type: ignore

    def get_size_statistics(self, incl_vanished: bool = True) -> dict[str, float]:
        """determine size statistics of the current emulsion

        Args:
            incl_vanished (bool):
                Whether to include droplets with vanished radii

        Returns:
            dict: a dictionary with various size statistics
        """
        if len(self) == 0:
            return {
                "count": 0,
                "radius_mean": math.nan,
                "radius_std": math.nan,
                "volume_mean": math.nan,
                "volume_std": math.nan,
            }

        if incl_vanished:
            radii = [droplet.radius for droplet in self]
            volumes = [droplet.volume for droplet in self]
        else:
            radii = [droplet.radius for droplet in self if droplet.radius > 0]
            volumes = [droplet.volume for droplet in self if droplet.radius > 0]

        return {
            "count": len(radii),
            "radius_mean": np.mean(radii),
            "radius_std": np.std(radii),
            "volume_mean": np.mean(volumes),
            "volume_std": np.std(volumes),
        }

    @plot_on_axes()
    def plot(
        self,
        ax,
        field: ScalarField | None = None,
        image_args: dict[str, Any] | None = None,
        repeat_periodically: bool = True,
        color_value: Callable | None = None,
        cmap=None,
        norm=None,
        colorbar: bool | str = True,
        **kwargs,
    ) -> PlotReference:
        """plot the current emulsion together with a corresponding field

        If the emulsion is defined in a 3d geometry, only a projection on the first two
        axes is shown.

        Args:
            {PLOT_ARGS}
            field (:class:`pde.fields.scalar.ScalarField`):
                provides the phase field that is shown as a background
            image_args (dict):
                additional arguments determining how the phase field in the
                background is plotted. Acceptable arguments are described in
                :func:`~pde.fields.base.FieldBase.plot`.
            repeat_periodically (bool):
                flag determining whether droplets are shown on both sides of
                periodic boundary conditions. This option can slow down plotting
            color_value (callable):
                Function used to determine the color of a droplet. The function is
                called with individual droplet objects and must return a single scalar
                value, which is then mapped to a color using the colormap given by
                `cmap` and a suitable normalization given by `norm`.
            cmap (str or :class:`~matplotlib.colors.Colormap`):
                The colormap used to map normalized data values to RGBA colors.
            norm (:class:`~matplotlib.colors.Normalize`):
                The normalizing object which scales data, typically into the interval
                [0, 1]. If None, norm defaults to a `colors.Normalize` object which
                maps the range of values obtained from `color_value` to [0, 1].
            colorbar (bool or str):
                Determines whether a colorbar is shown when `color_value` is supplied.
                If a string is given, it is used as a label for the colorbar.
            **kwargs:
                Additional keyword arguments are passed to the function creating the
                patch that represents the droplet. For instance, to only draw the
                outlines of the droplets, you may need to supply `fill=False`.

        Returns:
            :class:`~pde.tools.plotting.PlotReference`: Information about the plot
        """
        if len(self) == 0:
            # empty emulsions can be plotted in all dimensions :)
            return PlotReference(ax, [], {})
        if self.dim is None or self.dim <= 1:
            raise NotImplementedError(
                f"Plotting emulsions in {self.dim} dimensions is not implemented."
            )
        elif self.dim > 2:
            if Emulsion._show_projection_warning:
                logger = logging.getLogger(self.__class__.__name__)
                logger.warning("A projection on the first two axes is shown.")
                Emulsion._show_projection_warning = False

        # plot background and determine bounds for the droplets
        if field is not None:
            # plot the phase field and use its bounds
            if image_args is None:
                image_args = {}
            field.plot(kind="image", ax=ax, **image_args)
            ax.autoscale(False)  # fix image bounds to phase field
            grid = field.grid
            if isinstance(grid, CartesianGrid):
                # determine the bounds from the (2d) grid
                bounds = grid.axes_bounds
            else:
                # determine the bounds from the emulsion data itself
                bounds = self.bbox.bounds
        else:
            # determine the bounds from the emulsion data itself
            grid = None
            bounds = self.bbox.bounds

        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        ax.set_aspect("equal")

        # determine non-vanishing droplets
        drops_finite = [droplet for droplet in self if droplet.radius > 0]

        # determine the color of all droplets
        if color_value is not None:
            import matplotlib.pyplot as plt

            # determine the scalar values associated with all droplets
            values = np.array([color_value(droplet) for droplet in drops_finite])

            # and map them to colors
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colors: list | np.ndarray = mapper.to_rgba(values)

            if kwargs.pop("color", None) is not None:
                logger = logging.getLogger(self.__class__.__name__)
                logger.warning("`color` is overwritten by `color_value`.")

        else:
            colors = [kwargs.pop("color", None)] * len(drops_finite)

        # get patches representing all droplets
        if grid is None or not repeat_periodically:
            # plot only the droplets themselves
            patches = [
                droplet._get_mpl_patch(dim=2, color=color, **kwargs)
                for droplet, color in zip(drops_finite, colors)
            ]
        else:
            # plot droplets also in their mirror positions
            patches = []
            for droplet, color in zip(drops_finite, colors):
                for p in grid.iter_mirror_points(
                    droplet.position, with_self=True, only_periodic=True
                ):
                    # create copy with changed position
                    d = droplet.copy(position=p)
                    patches.append(d._get_mpl_patch(dim=2, color=color, **kwargs))

        # add all patches as a collection
        import matplotlib as mpl

        # the zorder needs to be set on the collection level to have any effect
        col_args = {}
        if "zorder" in kwargs:
            col_args["zorder"] = kwargs["zorder"]
        coll = mpl.collections.PatchCollection(patches, match_original=True, **col_args)
        ax.add_collection(coll)

        # add colorbar if requested
        if color_value is not None and colorbar:
            from pde.tools.plotting import add_scaled_colorbar

            label = colorbar if isinstance(colorbar, str) else ""
            add_scaled_colorbar(mapper, ax=ax, label=label)

        parameters = {
            "repeat_periodicially": repeat_periodically,
        }
        return PlotReference(ax, coll, parameters)


class EmulsionTimeCourse:
    """represents emulsions as a function of time"""

    def __init__(
        self,
        emulsions: Iterable[Emulsion] | None = None,
        times: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        """
        Args:
            emulsions (list): List of emulsions that describe this time course
            times (list): Times associated with the emulsions
        """
        if isinstance(emulsions, EmulsionTimeCourse):
            # extract data from given object; ignore `times`
            times = emulsions.times
            emulsions = emulsions.emulsions

        self.emulsions: list[Emulsion] = []
        self.times: list[float] = []

        # add all emulsions
        if emulsions is not None:
            for e in emulsions:
                self.append(Emulsion(e))

        # add all times
        if times is None:
            self.times = list(range(len(self.emulsions)))
        else:
            self.times = list(times)

        if len(self.times) != len(self.emulsions):
            raise ValueError("Lists of emulsions and times must have same length")

    def append(
        self, emulsion: Emulsion, time: float | None = None, copy: bool = True
    ) -> None:
        """add an emulsion to the list

        Args:
            emulsions (Emulsion):
                An :class:`Emulsion` instance that is added to the time course
            time (float):
                The time point associated with this emulsion
            copy (bool):
                Whether to copy the emulsion
        """
        emulsion = Emulsion(emulsion)  # make sure this is an emulsion

        # add the emulsion
        if copy:
            emulsion = emulsion.copy()
        self.emulsions.append(emulsion)

        if time is None:
            time = 0 if len(self.times) == 0 else self.times[-1] + 1
        self.times.append(time)

    def clear(self) -> None:
        """removes all data stored in this instance"""
        self.emulsions = []
        self.times = []

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(emulsions=Array({len(self)}), "
            f"times=Array({len(self)}))"
        )

    def __len__(self):
        return len(self.times)

    def __getitem__(self, key: int | slice):
        """return the information for the given index"""
        result = self.emulsions.__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(emulsions=result, times=self.times[key])
        else:
            return result

    def __iter__(self) -> Iterator[Emulsion]:
        """iterate over the emulsions"""
        return iter(self.emulsions)

    def items(self) -> Iterator[tuple[float, Emulsion]]:
        """iterate over all times and emulsions, returning them in pairs"""
        return zip(self.times, self.emulsions)

    def __eq__(self, other):
        """determine whether two EmulsionTimeCourse instance are equal"""
        return self.times == other.times and self.emulsions == other.emulsions

    @classmethod
    def from_storage(
        cls,
        storage: StorageBase,
        *,
        num_processes: int | Literal["auto"] = 1,
        refine: bool = False,
        progress: bool | None = None,
        **kwargs,
    ) -> EmulsionTimeCourse:
        r"""create an emulsion time course from a stored phase field

        Args:
            storage (:class:`~pde.storage.base.StorageBase`):
                The phase fields for many time instances
            refine (bool):
                Flag determining whether the droplet properties should be refined
                using fitting. This is a potentially slow procedure.
            num_processes (int or "auto"):
                Number of processes used for the refinement. If set to "auto", the
                number of processes is choosen automatically.
            progress (bool):
                Whether to show the progress of the process. If `None`, the progress is
                only shown when `refine` is `True`. Progress bars are only shown for
                serial calculations (where `num_processes == 1`).
            \**kwargs:
                All other parameters are forwarded to the
                :meth:`~droplets.image_analysis.locate_droplets`.

        Returns:
            EmulsionTimeCourse: an instance describing the emulsion time course
        """
        from .image_analysis import locate_droplets

        if num_processes == 1:
            # obtain the emulsion data for all frames in this process

            if progress is None:
                progress = refine  # show progress only when refining by default

            emulsions: Iterable[Emulsion] = (
                locate_droplets(frame, refine=refine, **kwargs)
                for frame in display_progress(storage, enabled=progress)
            )

        else:
            # use multiprocessing to obtain emulsion data
            from concurrent.futures import ProcessPoolExecutor

            _get_emulsion: Callable[[Emulsion], Emulsion] = functools.partial(
                locate_droplets, refine=refine, **kwargs
            )

            max_workers = None if num_processes == "auto" else num_processes
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                emulsions = list(executor.map(_get_emulsion, storage))

        return cls(emulsions, times=storage.times)

    @classmethod
    def from_file(cls, path: str, progress: bool = True) -> EmulsionTimeCourse:
        """create emulsion time course by reading file

        Args:
            path (str):
                The path from which the data is read. This function assumes that the
                data was written as an HDF5 file using :meth:`to_file`.
            progress (bool):
                Whether to show the progress of the process in a progress bar

        Returns:
            EmulsionTimeCourse: an instance describing the emulsion time course
        """
        import h5py

        obj = cls()
        with h5py.File(path, "r") as fp:
            # load the actual emulsion data and iterate in the right order
            for key in display_progress(
                sorted(fp.keys()), total=len(fp), enabled=progress
            ):
                dataset = fp[key]
                obj.append(
                    Emulsion._from_hdf_dataset(dataset), time=dataset.attrs["time"]
                )
        return obj

    def to_file(self, path: str, info: InfoDict | None = None) -> None:
        """store data in hdf5 file

        The data can be read using the classmethod :meth:`EmulsionTimeCourse.from_file`.

        Args:
            path (str):
                The path to which the data is written as an HDF5 file.
            info (dict):
                Additional data stored alongside the droplet track list
        """
        import h5py

        with h5py.File(path, "w") as fp:
            # write the actual emulsion data
            for i, (time, emulsion) in enumerate(self.items()):
                dataset = emulsion._write_hdf_dataset(fp, f"time_{i:06d}")
                dataset.attrs["time"] = time

            # write additional information
            if info:
                for k, v in info.items():
                    fp.attrs[k] = json.dumps(v)

    def get_emulsion(self, time: float) -> Emulsion:
        """returns the emulsion clostest to a specific time point

        Args:
            time (float): The time point

        Returns:
            :class:`Emuslion`
        """
        idx = np.argmin(np.abs(np.asarray(self.times) - time))
        return self.emulsions[idx]

    @fill_in_docstring
    def tracker(
        self,
        interrupts: InterruptData = 1,
        filename: str | None = None,
        *,
        interval=None,
    ) -> DropletTracker:
        """return a tracker that analyzes emulsions during simulations

        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPTS}
            filename (str): determines where the EmulsionTimeCourse data is
                stored
        """
        from .trackers import DropletTracker  # @Reimport

        return DropletTracker(
            emulsion_timecourse=self,
            filename=filename,
            interrupts=interrupts,
            interval=interval,
        )


__all__ = ["Emulsion", "EmulsionTimeCourse"]
