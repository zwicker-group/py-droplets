"""
Classes that describe collections of droplets, i.e. emulsions, and their
temporal dynamics.


.. autosummary::
   :nosignatures:

   Emulsion
   EmulsionTimeCourse


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json
import logging
from typing import List  # @UnusedImport
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from pde.fields import ScalarField
from pde.grids.base import GridBase
from pde.grids.cartesian import CartesianGridBase
from pde.storage.base import StorageBase
from pde.tools.cuboid import Cuboid
from pde.tools.output import display_progress
from pde.tools.plotting import plot_on_axes
from pde.trackers.base import InfoDict
from pde.trackers.intervals import IntervalType

from .droplets import SphericalDroplet, droplet_from_data

if TYPE_CHECKING:
    from .trackers import DropletTracker  # @UnusedImport


DropletSequence = Union[Generator, Sequence[SphericalDroplet]]


class Emulsion(list):
    """ class representing a collection of droplets in a common system """

    _show_projection_warning: bool = True

    def __init__(
        self,
        droplets: DropletSequence = None,
        grid: Optional[GridBase] = None,
        copy: bool = True,
    ):
        """
        Args:
            droplets:
                A list or generator of instances of
                :class:`~droplets.droplets.SphericalDroplet`.
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the droplets are defined. This information can
                helpful to measure distances between droplets correctly.
            copy (bool, optional):
                Whether to make a copy of the droplet or not
        """
        # obtain grid of the emulsion
        if isinstance(droplets, Emulsion):
            self.grid: Optional[GridBase] = droplets.grid
            if grid is not None and self.grid != grid:
                raise ValueError("Emulsion grid is unequal to given grid")
        else:
            self.grid = grid

        # determine space dimension
        if grid is not None:
            self.dim: Optional[int] = grid.dim  # dimension of the space
        else:
            self.dim = None

        # add all droplets
        super().__init__()
        if droplets is not None:
            self.extend(droplets, copy=copy)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and self.grid == other.grid

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__ne__(other) or self.grid != other.grid

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()}, grid={self.grid})"

    def __add__(self, rhs):
        # return sum of Emulsions as an emulsion
        return Emulsion(list.__add__(self, rhs))

    def __getitem__(self, key: Union[int, slice]):
        # return result from extended slicing as Emulsion
        result = list.__getitem__(self, key)
        try:
            return Emulsion(result, grid=self.grid)
        except TypeError:
            return result

    def copy(self, min_radius: float = -1) -> "Emulsion":
        """return a copy of this emulsion

        Args:
            min_radius (float):
                The minimal radius of the droplets that are retained. Droplets
                with exactly min_radius are removed, so `min_radius == 0` can be
                used to filter vanished droplets.
        """
        droplets = [droplet.copy() for droplet in self if droplet.radius > min_radius]
        return self.__class__(
            droplets,
            grid=self.grid,
            copy=False,
        )

    def extend(  # type: ignore
        self,
        droplets: DropletSequence,
        copy: bool = True,
    ) -> None:
        """add many droplets to the emulsion

        Args:
            droplets (list): List of droplets to add to the emulsion
            copy (bool, optional): Whether to make a copy of the droplets or not
        """
        for droplet in droplets:
            self.append(droplet, copy=copy)

    def append(self, droplet: SphericalDroplet, copy: bool = True) -> None:
        """add a droplet to the emulsion

        Args:
            droplet (:class:`droplets.dropelts.SphericalDroplet`):
                Droplet to add to the emulsion
            copy (bool, optional):
                Whether to make a copy of the droplet or not
        """
        if self.dim is None:
            self.dim = droplet.dim
        elif self.dim != droplet.dim:
            raise ValueError(
                f"Cannot append droplet in dimension {droplet.dim} to emulsion in "
                f"dimension {self.dim}"
            )
        if copy:
            droplet = droplet.copy()
        super().append(droplet)

    @property
    def data(self) -> Optional[np.ndarray]:
        """:class:`~numpy.ndarray`: an array containing the data of the full emulsion

        This requires all droplets to be of the same class
        """
        if len(self) == 0:
            return None
        else:
            # emulsion contains at least one droplet
            classes = set(d.__class__ for d in self)
            if len(classes) > 1:
                raise TypeError(
                    "Emulsion data cannot be stored contiguously if it contains a "
                    f"multiple of droplet classes: "
                    + ", ".join(c.__name__ for c in classes)
                )
            return np.array([d.data for d in self])

    def get_linked_data(self) -> np.ndarray:
        """link the data of all droplets in a single array

        Returns:
            :class:`~numpy.ndarray`: The array containing all droplet data. If entries in
                this array are modified, it will be reflected in the droplets.
        """
        if len(self) == 0:
            data = np.empty(0)
        else:
            assert self.data is not None
            data = self.data  # create an array with all the droplet data
            # link back to droplets
            for i, d in enumerate(self):
                d.data = data[i]
        return data

    @classmethod
    def _from_hdf_dataset(cls, dataset, grid: Optional[GridBase] = None) -> "Emulsion":
        """construct an emulsion by reading data from an hdf5 dataset

        Args:
            dataset:
                an HDF5 dataset from which the data of the emulsion is read
            grid (:class:`pde.grids.base.GridBase`):
                The grid on which the droplets are defined. This information is required
                to measure distances between droplets.
        """
        # there are values, so the emulsion is not empty
        droplet_class = dataset.attrs["droplet_class"]
        if droplet_class == "None":
            droplets: List[SphericalDroplet] = []
        else:
            droplets = [
                droplet_from_data(droplet_class, data)  # type: ignore
                for data in dataset
            ]
        return cls(droplets, grid=grid, copy=False)

    @classmethod
    def from_file(cls, filename: str) -> "Emulsion":
        """create emulsion by reading file

        Args:
            filename (str): Name of the file to read emulsion from
        """
        import h5py

        with h5py.File(filename, "r") as fp:
            if len(fp) != 1:
                raise RuntimeError(f"Multiple emulsions found in file `{filename}`")

            # read grid
            if "grid" in fp.attrs:
                grid: Optional[GridBase] = GridBase.from_state(fp.attrs["grid"])
            else:
                grid = None

            # read the actual droplet data
            dataset = fp[list(fp.keys())[0]]  # retrieve the only dataset
            obj = cls._from_hdf_dataset(dataset, grid)

        return obj

    def _write_hdf_dataset(self, hdf_path, key: str = "emulsion"):
        """ write data to a given hdf5 path `hdf_path` """
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

    def to_file(self, filename: str) -> None:
        """store data in hdf5 file

        Args:
            filename (str): Name of the file to write emulsion to
        """
        import h5py

        with h5py.File(filename, "w") as fp:
            # write the grid data
            if self.grid is not None:
                fp.attrs["grid"] = self.grid.state_serialized
            # write actual droplet data
            self._write_hdf_dataset(fp)

    @property
    def interface_width(self) -> Optional[float]:
        """float: the average interface width across all droplets

        This averages the interface widths of the individual droplets weighted
        by their surface area, i.e., the amount of interface
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
        """ :class:`Cuboid`: bounding box of the emulsion """
        if len(self) == 0:
            raise RuntimeError("Bounding box of empty emulsion is undefined")
        return sum((droplet.bbox for droplet in self[1:]), self[0].bbox)

    def get_phasefield(
        self, grid: GridBase = None, label: Optional[str] = None
    ) -> ScalarField:
        """create a phase field representing a list of droplets

        Args:
            grid (:class:`pde.grids.base.GridBase`):
                The grid on which the phase field is created. If omitted, the
                grid associated with the emulsion is used.
            label (str):
                Optional label for the returned scalar field

        Returns:
            :class:`~pde.fields.scalar.ScalarField`: the actual phase field
        """
        if grid is None:
            grid = self.grid
        if grid is None:
            raise RuntimeError("Grid needs to be specified")

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
            min_radius (float): The minimal radius of the droplets that are
                retained. Droplets with exactly min_radius are removed, so
                `min_radius == 0` can be used to filter vanished droplets. The
                default value does not remove any droplets
        """
        for i in reversed(range(len(self))):
            if self[i].radius <= min_radius:
                self.pop(i)

    def get_pairwise_distances(self, subtract_radius: bool = False) -> np.ndarray:
        """return the pairwise distance between droplets

        Args:
            subtract_radius (bool): determines whether the distance is measured
                from interface to interface (for round droplets) or center to
                center.

        Returns:
            :class:`~numpy.ndarray`: a matrix with the distances between all droplets
        """
        if self.grid is None:

            def get_distance(p1, p2):
                """ helper function calculating the distance between points """
                return np.linalg.norm(p1 - p2)

        else:
            get_distance = self.grid.distance_real

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

        Warning: Nearest neighbors are defined by comparing the distances
        between the centers of the droplets, not their surfaces.

        Args:
            subtract_radius (bool): Determines whether to subtract the radius
                from the distance, i.e., whether to return the distance between
                the surfaces instead of the positions

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
        tree = KDTree(positions)
        dist, index = tree.query(positions, 2)

        if subtract_radius:
            return dist[:, 1] - self.data["radius"][index].sum(axis=1)  # type: ignore
        else:
            return dist[:, 1]  # type: ignore

    def remove_overlapping(self, min_distance: float = 0) -> None:
        """remove all droplets that are overlapping.

        If a pair of overlapping droplets was found, the smaller one of these
        is removed from the current emulsion. This method modifies the emulsion
        in place and thus does not return anything.

        Args:
            min_distance (float): The minimal distance droplets need to be
                apart. The default value of 0 corresponds to just remove
                overlapping droplets. Larger values ensure that droplets keep a
                distance, while negative values allow for some overlap.
        """
        # filter duplicates until there are none left
        dists = self.get_pairwise_distances(subtract_radius=True)
        np.fill_diagonal(dists, np.inf)

        while len(dists) > 1:
            # find minimal distance
            x, y = np.unravel_index(np.argmin(dists), dists.shape)
            if dists[x, y] < min_distance:
                # droplets overlap -> remove the smaller one
                if self[x].radius > self[y].radius:
                    self.pop(y)
                    dists = np.delete(np.delete(dists, y, 0), y, 1)
                else:
                    self.pop(x)
                    dists = np.delete(np.delete(dists, x, 0), x, 1)
            else:
                break

    @property
    def total_droplet_volume(self) -> float:
        """ float: the total volume of all droplets """
        return sum(droplet.volume for droplet in self)

    def get_size_statistics(self) -> Dict[str, float]:
        """determine size statistics of the current emulsion

        Returns:
            dict: a dictionary with various size statistics
        """
        if len(self) == 0:
            return {
                "count": 0,
                "radius_mean": np.nan,
                "radius_std": np.nan,
                "volume_mean": np.nan,
                "volume_std": np.nan,
            }

        radii = [droplet.radius for droplet in self]
        volumes = [droplet.volume for droplet in self]
        return {
            "count": len(self),
            "radius_mean": np.mean(radii),
            "radius_std": np.std(radii),
            "volume_mean": np.mean(volumes),
            "volume_std": np.std(volumes),
        }

    @plot_on_axes()
    def plot(
        self,
        ax,
        field: ScalarField = None,
        image_args: Dict[str, Any] = None,
        repeat_periodically: bool = True,
        **kwargs,
    ):
        """plot the current emulsion together with a corresponding field

        If the emulsion is defined in a 3d geometry, only a projection on the first two
        axes is shown.

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                The axes in which the background is shown
            field (:class:`pde.fields.scalar.ScalarField`):
                provides the phase field that is shown as a background
            image_args (dict):
                additional arguments determining how the phase field in the
                background is plotted. Acceptable arguments are described in
                :func:`~pde.fields.base.FieldBase.plot`.
            repeat_periodically (bool):
                flag determining whether droplets are shown on both sides of
                periodic boundary conditions. This option can slow down plotting
            **kwargs:
                Additional keyword arguments are passed to the function creating the
                patch that represents the droplet. For instance, to only draw the
                outlines of the droplets, you may need to supply `fill=False`.
        """
        if self.dim is None or self.dim <= 1:
            raise NotImplementedError(
                f"Plotting emulsions in {self.dim} dimensions is not implemented."
            )
        elif self.dim > 2:
            if Emulsion._show_projection_warning:
                logger = logging.getLogger(self.__class__.__name__)
                logger.warning("A projection on the first two axes is shown.")
                Emulsion._show_projection_warning = False

        grid_compatible = (
            self.grid is None or field is None or self.grid.compatible_with(field.grid)
        )
        if not grid_compatible:
            raise ValueError("Emulsion grid is not compatible with field grid")
        grid = self.grid

        # plot background and determine bounds for the droplets
        if field is not None:
            # plot the phase field and use its bounds
            if image_args is None:
                image_args = {}
            field.plot(kind="image", ax=ax, **image_args)
            ax.autoscale(False)  # fix image bounds to phase field
            if grid is None:
                grid = field.grid

        else:
            if isinstance(grid, CartesianGridBase):
                # determine the bounds from the (2d) grid
                bounds = grid.axes_bounds
            else:
                # determine the bounds from the emulsion data itself
                bounds = self.bbox.bounds
            ax.set_xlim(*bounds[0])
            ax.set_ylim(*bounds[1])
            ax.set_aspect("equal")

        # get patches representing all droplets
        if grid is None or not repeat_periodically:
            # plot only the droplets themselves
            patches = [droplet._get_mpl_patch(dim=2, **kwargs) for droplet in self]
        else:
            # plot droplets also in their mirror positions
            patches = []
            for droplet in self:
                for p in grid.iter_mirror_points(
                    droplet.position, with_self=True, only_periodic=True
                ):
                    # create copy with changed position
                    d = droplet.copy(position=p)
                    patches.append(d._get_mpl_patch(dim=2, **kwargs))

        # add all patches as a collection
        import matplotlib as mpl

        coll = mpl.collections.PatchCollection(patches, match_original=True)
        ax.add_collection(coll)


class EmulsionTimeCourse:
    """ represents emulsions as a function of time """

    def __init__(self, emulsions=None, times=None):
        """
        Args:
            emulsions (list): List of emulsions that describe this time course
            times (list): Times associated with the emulsions
        """
        if isinstance(emulsions, EmulsionTimeCourse):
            # extract data from given object; ignore `times`
            times = emulsions.times
            self.grid = emulsions.grid
            emulsions = emulsions.emulsions
        else:
            self.grid = None

        self.emulsions = []
        self.times = []

        # add all emulsions
        if emulsions:
            for e in emulsions:
                self.append(Emulsion(e))

        # add all times
        if times is not None:
            self.times = list(times)

        if len(self.times) != len(self.emulsions):
            raise ValueError(
                "The list of emulsions and the list of times need "
                "to have the same length"
            )

    def append(
        self, emulsion: Emulsion, time: Optional[float] = None, copy: bool = True
    ) -> None:
        """add an emulsion to the list

        Args:
            emulsions (Emulsion): An :class:`Emulsion` instance that is added
                to the time course
            time (float): The time point associated with this emulsion
            copy (bool): Whether to copy the emulsion
        """
        emulsion = Emulsion(emulsion)  # make sure this is an emulsion

        # check grid consistency
        if self.grid is None:
            self.grid = emulsion.grid
        elif emulsion.grid and not self.grid.compatible_with(emulsion.grid):
            raise ValueError(
                "Grid of the EmulsionTimeCourse is not compatible with the grid of the "
                f"emulsion to be added. ({self.grid} != {emulsion.grid})"
            )
        # add the emulsion
        if copy:
            emulsion = emulsion.copy()
        self.emulsions.append(emulsion)

        if time is None:
            time = 0 if len(self.times) == 0 else self.times[-1] + 1
        self.times.append(time)

    def clear(self) -> None:
        """ removes all data stored in this instance """
        self.emulsions = []
        self.times = []

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(emulsions=Array({len(self)}), "
            f"times=Array({len(self)}), grid={self.grid})"
        )

    def __len__(self):
        return len(self.times)

    def __getitem__(self, key: Union[int, slice]):
        """ return the information for the given index """
        result = self.emulsions.__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(emulsions=result, times=self.times[key])
        else:
            return result

    def __iter__(self) -> Iterator[Tuple[float, Emulsion]]:
        """ iterate over the emulsions """
        return iter(self.emulsions)

    def items(self):
        """ iterate over all times and emulsions, returning them in pairs """
        return zip(self.times, self.emulsions)

    def __eq__(self, other):
        """ determine whether two EmulsionTimeCourse instance are equal """
        grids_equal = self.grid is None or other.grid is None or self.grid == other.grid
        return (
            grids_equal
            and self.times == other.times
            and self.emulsions == other.emulsions
        )

    @classmethod
    def from_storage(
        cls, storage: StorageBase, refine: bool = False, progress: bool = None, **kwargs
    ) -> "EmulsionTimeCourse":
        r"""create an emulsion time course from a stored phase field

        Args:
            storage (:class:`~pde.storage.base.StorageBase`):
                The phase fields for many time instances
            refine (bool):
                Flag determining whether the droplet properties should be refined
                using fitting. This is a potentially slow procedure.
            progress (bool):
                Whether to show the progress of the process. If `None`, the progress is
                only shown when `refine` is `True`.
            \**kwargs:
                All other parameters are forwarded to the
                :meth:`~droplets.image_analysis.locate_droplets`.

        Returns:
            EmulsionTimeCourse: an instance describing the emulsion time course
        """
        from .image_analysis import locate_droplets

        if progress is None:
            progress = refine  # show progress only when refining by default

        # obtain the emulsion data for all frames
        emulsions = (
            locate_droplets(frame, refine=refine, **kwargs)
            for frame in display_progress(storage, enabled=progress)
        )

        return cls(emulsions, times=storage.times)

    @classmethod
    def from_file(cls, filename: str, progress: bool = True) -> "EmulsionTimeCourse":
        """create emulsion time course by reading file

        Args:
            filename (str): The filename from which the emulsion is read
            progress (bool): Whether to show the progress of the process

        Returns:
            EmulsionTimeCourse: an instance describing the emulsion time course
        """
        import h5py

        obj = cls()
        with h5py.File(filename, "r") as fp:
            # read grid
            if "grid" in fp.attrs:
                grid: Optional[GridBase] = GridBase.from_state(fp.attrs["grid"])
            else:
                grid = None

            # load the actual emulsion data and iterate in the right order
            for key in display_progress(
                sorted(fp.keys()), total=len(fp), enabled=progress
            ):
                dataset = fp[key]
                obj.append(
                    Emulsion._from_hdf_dataset(dataset, grid),
                    time=dataset.attrs["time"],
                )
        return obj

    def to_file(self, filename: str, info: InfoDict = None) -> None:
        """store data in hdf5 file

        Args:
            filename (str): determines the location where the file is written
            info (dict): can be additional data stored alongside
        """
        import h5py

        with h5py.File(filename, "w") as fp:
            # write the actual emulsion data
            for i, (time, emulsion) in enumerate(self.items()):
                dataset = emulsion._write_hdf_dataset(fp, f"time_{i:06d}")
                dataset.attrs["time"] = time

            # write additional information
            if info:
                for k, v in info.items():
                    fp.attrs[k] = json.dumps(v)

            # write the grid data -> this might overwrite grid data that is
            # present in the info dictionary, but in normal cases these grids
            # should be identical, so we don't handle this case explicitly
            if self.grid is not None:
                fp.attrs["grid"] = self.grid.state_serialized

    def tracker(
        self,
        interval: Union[int, float, IntervalType] = 1,
        filename: Optional[str] = None,
    ) -> "DropletTracker":
        """return a tracker that analyzes emulsions during simulations

        Args:
            interval: Determines how often the tracker interrupts the
                simulation. Simple numbers are interpreted as durations measured
                in the simulation time variable. Alternatively, instances of
                :class:`~droplets.simulation.trackers.LogarithmicIntervals` and
                :class:`~droplets.simulation.trackers.RealtimeIntervals` might
                be given for more control.
            filename (str): determines where the EmulsionTimeCourse data is
                stored
        """
        from .trackers import DropletTracker  # @Reimport

        return DropletTracker(
            emulsion_timecourse=self, filename=filename, interval=interval
        )


__all__ = ["Emulsion", "EmulsionTimeCourse"]
