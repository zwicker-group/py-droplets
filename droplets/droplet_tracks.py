"""Classes representing the time evolution of droplets.

.. autosummary::
   :nosignatures:

   DropletTrack
   DropletTrackList

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Callable, Literal

import numpy as np
from numpy.lib import recfunctions as rfn
from scipy import ndimage
from scipy.spatial import distance

from pde.grids.base import GridBase
from pde.storage.base import StorageBase
from pde.tools.output import display_progress
from pde.tools.plotting import PlotReference, plot_on_axes
from pde.trackers.base import InfoDict

from .droplets import SphericalDroplet, droplet_from_data
from .emulsions import Emulsion, EmulsionTimeCourse

_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


def contiguous_true_regions(condition: np.ndarray) -> np.ndarray:
    """Finds contiguous True regions in the boolean array "condition".

    Inspired by http://stackoverflow.com/a/4495197/932593

    Args:
        condition (:class:`~numpy.ndarray`):
            A one-dimensional boolean array

    Returns:
        :class:`~numpy.ndarray`: A two-dimensional array where the first column
        is the start index of the region and the second column is the end index
    """
    if len(condition) == 0:
        return np.empty((0, 2), dtype=np.intc)

    # convert condition array to integer
    condition = np.asarray(condition, np.intc)

    # Find the indices of changes in "condition"
    d = np.diff(condition)
    idx = np.flatnonzero(d)

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]

    # Reshape the result into two columns
    return idx.reshape(-1, 2).astype(np.intc)  # type:ignore


class DropletTrack:
    """Information about a single droplet over multiple time steps."""

    def __init__(self, droplets=None, times=None):
        """
        Args:
            emulsions (list):
                List of emulsions that describe this time course
            times (list):
                Times associated with the emulsions
        """
        if isinstance(droplets, DropletTrack):
            # get data from given object
            times = droplets.times
            droplets = droplets.droplets

        self.droplets = []
        self.times = []

        # add all emulsions
        if droplets:
            for d in droplets:
                self.append(d)

        # add all times
        if times is not None:
            self.times = list(times)

        if len(self.times) != len(self.droplets):
            raise ValueError(
                "The lists of droplets and times need to have the same length"
            )

    def __repr__(self):
        """Human-readable representation of a droplet track."""
        class_name = self.__class__.__name__
        if len(self.times) == 0:
            return f"{class_name}([])"
        elif len(self.times) == 1:
            return f"{class_name}(time={self.start:g})"
        else:
            return f"{class_name}(timespan={self.start:g}..{self.end:g})"

    def __len__(self):
        """Number of time points."""
        return len(self.times)

    def __getitem__(self, key: int | slice):
        """Return the droplets identified by the given index/slice."""
        result = self.droplets.__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(droplets=result, times=self.times[key])
        else:
            return result

    def __eq__(self, other):
        """Determine whether two DropletTracks instance are equal."""
        return self.times == other.times and self.droplets == other.droplets

    @property
    def start(self) -> float:
        """float: first time point"""
        return self.times[0]  # type: ignore

    @property
    def end(self) -> float:
        """float: last time point"""
        return self.times[-1]  # type: ignore

    @property
    def duration(self) -> float:
        """float: total duration of the track"""
        if len(self.times) > 0:
            return self.end - self.start
        else:
            return 0

    @property
    def first(self) -> SphericalDroplet:
        """SphericalDroplet: first droplet instance"""
        return self.droplets[0]  # type: ignore

    @property
    def last(self) -> SphericalDroplet:
        """SphericalDroplet: last droplet instance"""
        return self.droplets[-1]  # type: ignore

    @property
    def dim(self) -> int | None:
        """Return the space dimension of the droplets."""
        try:
            return self.last.dim
        except IndexError:
            return None

    @property
    def data(self) -> np.ndarray | None:
        """:class:`~numpy.ndarray`: an array containing the data of the full track."""
        if len(self) == 0:
            return None
        else:
            d0 = self.first
            dtype = [("time", "f8")] + d0.data.dtype.descr
            result = np.empty(len(self), dtype=dtype)
            for i in range(len(self)):
                result[i] = (self.times[i],) + self.droplets[i].data.tolist()
            return result  # type:ignore

    def __iter__(self):
        """Iterate over all droplets."""
        return iter(self.droplets)

    def items(self):
        """Iterate over all times and droplets, returning them in pairs."""
        return zip(self.times, self.droplets)

    def append(self, droplet: SphericalDroplet, time: float | None = None) -> None:
        """Append a new droplet with a time code.

        Args:
            droplet (:class:`droplets.droplets.SphericalDroplet`):
                The droplet to append
            time (float, optional):
                The associated time point
        """
        if self.dim is not None and droplet.dim != self.dim:
            raise ValueError(
                "Space dimension of added droplet must match the DropletTrack "
                f" ({droplet.dim} != {self.dim})"
            )

        # add the emulsion
        self.droplets.append(droplet.copy())
        if time is None:
            time = 0 if len(self.times) == 0 else self.times[-1] + 1
        self.times.append(time)

    def get_position(self, time: float) -> np.ndarray:
        """:class:`~numpy.ndarray`: returns the droplet position at a specific time."""
        try:
            idx = self.times.index(time)
        except AttributeError:
            # assume that self.times is a numpy array
            idx = np.nonzero(self.times == time)[0][0]
        return self.droplets[idx].position  # type: ignore

    def get_trajectory(
        self, smoothing: float = 0, *, attribute: str = "position"
    ) -> np.ndarray:
        """Return a the time-evolution of a droplet attribute (e.g., the position)

        Args:
            smoothing (float):
                Determines the scale for some gaussian smoothing of the trajectory.
                The default value of zero disables smoothing.
            attribute (str):
                The attribute to consider (default: "position").

        Returns:
            :class:`~numpy.ndarray`: An array giving the position of the droplet at each
                time instance
        """
        trajectory = np.array([getattr(d, attribute) for d in self.droplets])
        if smoothing:
            ndimage.gaussian_filter1d(
                trajectory, output=trajectory, sigma=smoothing, axis=0, mode="nearest"
            )
        return trajectory  # type:ignore

    def get_radii(self, smoothing: float = 0) -> np.ndarray:
        """:class:`~numpy.ndarray`: returns the droplet radius for each time point.

        Args:
            smoothing (float):
                Determines the length scale for some gaussian smoothing of the
                trajectory. The default value of zero disables smoothing.
        """
        return self.get_trajectory(smoothing, attribute="radius")

    def get_volumes(self, smoothing: float = 0) -> np.ndarray:
        """:class:`~numpy.ndarray`: returns the droplet volume for each time point.

        Args:
            smoothing (float):
                Determines the volume scale for some gaussian smoothing of the
                trajectory. The default value of zero disables smoothing.
        """
        return self.get_trajectory(smoothing, attribute="volume")

    def time_overlaps(self, other: DropletTrack) -> bool:
        """Determine whether two DropletTrack instances overlaps in time.

        Args:
            other (:class:`DropletTrack`):
                The other droplet track

        Returns:
            bool: True when both tracks contain droplets at the same time step
        """
        s0, s1 = self.start, self.end
        o0, o1 = other.start, other.end
        return s0 <= o1 and o0 <= s1

    @classmethod
    def _from_hdf_dataset(cls, dataset) -> DropletTrack:
        """Construct a droplet track by reading data from an hdf5 dataset.

        Args:
            dataset:
                A HDF5 dataset from which the data of the droplet track is read
        """
        # there are values, so the emulsion is not empty
        droplet_class = dataset.attrs["droplet_class"]
        obj = cls()
        if droplet_class == "None":
            return obj
        else:
            # separate time from the data set
            times = dataset["time"]
            droplet_data = rfn.rec_drop_fields(dataset, "time")
            for time, data in zip(times, droplet_data):
                droplet = droplet_from_data(droplet_class, data)
                obj.append(droplet, time=time)  # type: ignore

        return obj

    @classmethod
    def from_file(cls, path: str) -> DropletTrack:
        """Create droplet track by reading from file.

        Args:
            path (str):
                The path from which the data is read. This function assumes that the
                data was written as an HDF5 file using :meth:`to_file`.
        """
        import h5py

        with h5py.File(path, "r") as fp:
            if len(fp) != 1:
                raise RuntimeError(
                    f"Multiple droplet tracks found in file {path}. Did you mean to "
                    "load a DropletTrackList instead?"
                )
            dataset = fp[list(fp.keys())[0]]  # retrieve the only dataset
            obj = cls._from_hdf_dataset(dataset)

        return obj

    def _write_hdf_dataset(self, hdf_path, key: str = "droplet_track"):
        """Write data to a given hdf5 path `hdf_path`"""
        if self:
            # emulsion contains at least one droplet
            dataset = hdf_path.create_dataset(key, data=self.data)
            # self.data ensures that there is only one droplet class
            dataset.attrs["droplet_class"] = self[0].__class__.__name__

        else:
            # create empty dataset to indicate empty emulsion
            dataset = hdf_path.create_dataset(key, shape=())
            dataset.attrs["droplet_class"] = "None"

        return dataset

    def to_file(self, path: str, info: InfoDict | None = None) -> None:
        """Store data in hdf5 file.

        The data can be read using the classmethod :meth:`DropletTrack.from_file`.

        Args:
            path (str):
                The path to which the data is written as an HDF5 file.
            info (dict):
                Additional data stored alongside the droplet track list
        """
        import h5py

        with h5py.File(path, "w") as fp:
            self._write_hdf_dataset(fp)

            # write additional information
            if info:
                for k, v in info.items():
                    fp.attrs[k] = json.dumps(v)

    @plot_on_axes()
    def plot(
        self,
        attribute: str = "radius",
        smoothing: float = 0,
        t_max: float | None = None,
        ax=None,
        **kwargs,
    ) -> PlotReference:
        """Plot the time evolution of the droplet.

        Args:
            attribute (str):
                The attribute to plot. Typical values include `radius` and `volume`, but
                others might be defined on the droplet class.
            smoothing (float):
                Determines the scale for some gaussian smoothing of the trajectory.
                The default value of zero disables smoothing.
            {PLOT_ARGS}
            **kwargs:
                All remaining parameters are forwarded to the `ax.plot` method. For
                example, passing `color=None`, will use different colors for different
                droplets.

        Returns:
            :class:`~pde.tools.plotting.PlotReference`: Information about the plot
        """
        if len(self.times) == 0:
            return PlotReference(ax, None)

        if attribute in {"radius", "radii"}:
            data = self.get_radii()
            ylabel = "Radius"
        elif attribute in {"volume", "volumes"}:
            data = self.get_volumes()
            ylabel = "Volume"
        else:
            data = self.get_trajectory(smoothing=smoothing, attribute=attribute)
            ylabel = attribute.capitalize()

        if t_max is not None and len(self.times) >= 2 and self.times[-1] < t_max:
            dt = self.times[-1] - self.times[-2]
            times = np.r_[self.times, self.times[-1] + dt]
            data = np.r_[data, 0]
        else:
            times = self.times

        (line,) = ax.plot(times, data, **kwargs)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        return PlotReference(ax, line, {"attribute": attribute})

    @plot_on_axes()
    def plot_positions(
        self, grid: GridBase | None = None, arrow: bool = True, ax=None, **kwargs
    ) -> PlotReference:
        """Plot the droplet track.

        Args:
            grid (GridBase, optional):
                The grid on which the droplets are defined. If given, periodic boundary
                conditions can be respected in the plotting.
            arrow (bool, optional):
                Flag determining whether an arrow head is shown to indicate the
                direction of the droplet drift.
            {PLOT_ARGS}
            **kwargs:
                Additional keyword arguments are passed to the matplotlib plot
                function to affect the appearance. For example, passing `color=None`,
                will use different colors for different droplets.

        Returns:
            :class:`~pde.tools.plotting.PlotReference`: Information about the plot
        """
        if len(self.times) == 0:
            return PlotReference(ax, None, {"arrow": arrow})

        if self.dim != 2:
            raise NotImplementedError("Plotting is only implemented for 2d grids")

        # obtain droplet positions as a function of time
        xy = self.get_trajectory()

        if grid is None:
            # simply plot the trajectory
            cx, cy = xy[:, 0], xy[:, 1]
            (line,) = ax.plot(cx, cy, **kwargs)

        else:
            # use the grid to detect wrapping around
            segments = []
            for p1, p2 in zip(xy[:-1], xy[1:]):
                dist_direct = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                dist_real = grid.distance(p1, p2, coords="cartesian")
                close = np.isclose(dist_direct, dist_real)
                segments.append(close)

            # plot the individual segments
            line, cx = None, []  # type: ignore
            for s, e in contiguous_true_regions(np.array(segments)):
                if line is None:
                    color = kwargs.get("color", "k")
                else:
                    color = line.get_color()  # ensure colors stays the same

                cx, cy = xy[s : e + 1, 0], xy[s : e + 1, 1]
                (line,) = ax.plot(cx, cy, color=color)

        if arrow and len(cx) >= 2:
            # add arrow head to last segment
            size = min(sum(ax.get_xlim()), sum(ax.get_ylim()))
            ax.arrow(
                cx[-2],
                cy[-2],
                cx[-1] - cx[-2],
                cy[-1] - cy[-2],
                head_width=0.02 * size,
                color=line.get_color(),
            )

        return PlotReference(ax, None, {"arrow": arrow})


class DropletTrackList(list):
    """A list of instances of :class:`DropletTrack`"""

    def __getitem__(self, key: int | slice):  # type: ignore
        """Return the droplets identified by the given index/slice."""
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(result)
        else:
            return result

    @classmethod
    def from_emulsion_time_course(
        cls,
        time_course: EmulsionTimeCourse,
        *,
        method: Literal["distance", "overlap"] = "overlap",
        grid: GridBase | None = None,
        progress: bool = False,
        **kwargs,
    ) -> DropletTrackList:
        r"""Obtain droplet tracks from an emulsion time course.

        Args:
            time_course (:class:`droplets.emulsions.EmulsionTimeCourse`):
                A collection of temporally arranged emulsions
            method (str):
                The method used for tracking droplet identities. Possible methods are
                "overlap" (adding droplets that overlap with those in previous frames)
                and "distance" (matching droplets to minimize center-to-center
                distances).
            grid (:class:`~pde.grids.base.GridBase`):
                The grid on which the droplets are defined, which is necessary if
                periodic boundary conditions should be respected for measuring distances
            progress (bool):
                Whether to show the progress of the process.
            **kwargs:
                Additional parameters for the tracking algorithm. Currently, one can
                only specify a maximal distance (using `max_dist`) for the "distance"
                method.

        Returns:
            :class:`DropletTrackList`: the resulting droplet tracks
        """
        # get tracks, i.e. clearly overlapping droplets
        tracks = cls()

        # determine the tracking method
        if method == "overlap":
            # track droplets by their physical overlap

            def match_tracks(
                emulsion: Emulsion, tracks_alive: list[DropletTrack], time: float
            ) -> None:
                """Helper function adding emulsions to the tracks."""
                found_multiple_overlap = False
                for droplet in emulsion:
                    # determine which old tracks could be extended
                    overlaps: list[DropletTrack] = []
                    for track in tracks_alive:
                        if track.last.overlaps(droplet, grid=grid):
                            overlaps.append(track)

                    if len(overlaps) == 1:
                        overlaps[0].append(droplet, time=time)
                    else:
                        if len(overlaps) > 1:
                            found_multiple_overlap = True
                        tracks.append(DropletTrack(droplets=[droplet], times=[time]))

                if found_multiple_overlap:
                    _logger.debug("Found multiple overlapping droplet(s) at t=%g", time)

        elif method == "distance":
            # track droplets by their physical distance

            max_dist = kwargs.pop("max_dist", np.inf)

            def match_tracks(
                emulsion: Emulsion, tracks_alive: list[DropletTrack], time: float
            ) -> None:
                """Helper function adding emulsions to the tracks."""
                added = set()

                # calculate the distance between droplets
                if tracks_alive:
                    if grid is None:
                        metric: str | Callable = "euclidean"
                    else:
                        metric = functools.partial(grid.distance, coords="cartesian")
                    points_prev = [track.last.position for track in tracks_alive]
                    points_now = [droplet.position for droplet in emulsion]
                    dists = distance.cdist(points_prev, points_now, metric=metric)

                    # impose a cutoff distance
                    dists[dists > max_dist] = np.inf

                    # add all matching droplets
                    while True:
                        i, j = np.unravel_index(np.argmin(dists), dists.shape)
                        if np.isinf(dists[i, j]):
                            break  # no more matches
                        added.add(j)
                        tracks_alive[i].append(emulsion[j], time=time)  # type: ignore
                        dists[i, :] = np.inf
                        dists[:, j] = np.inf

                # add droplets that have not been matched
                for i, droplet in enumerate(emulsion):  # type: ignore
                    if i not in added:
                        tracks.append(DropletTrack(droplets=[droplet], times=[time]))

        else:
            raise ValueError("Unknown tracking method `%s`", method)

        # check kwargs
        if kwargs:
            _logger.warning("Unused keyword arguments: %s", kwargs)

        # add all emulsions successively using the given algorithm
        t_last = None
        for t, emulsion in display_progress(
            time_course.items(), total=len(time_course), enabled=progress
        ):
            # determine tracks from the last frame that have not yet been extended
            tracks_alive = [track for track in tracks if track.end == t_last]
            # match all tracks with the current emulsion
            match_tracks(emulsion, tracks_alive, time=t)
            t_last = t

        return tracks

    @classmethod
    def from_storage(
        cls,
        storage: StorageBase,
        *,
        method: Literal["distance", "overlap"] = "overlap",
        refine: bool = False,
        num_processes: int | Literal["auto"] = 1,
        progress: bool | None = None,
    ) -> DropletTrackList:
        r"""Obtain droplet tracks from stored scalar field data.

        This method first determines an emulsion time course and than collects tracks by
        tracking droplets.

        Args:
            storage (:class:`~pde.storage.base.StorageBase`):
                The phase fields for many time instances
            method (str):
                The method used for tracking droplet identities. Possible methods are
                "overlap" (adding droplets that overlap with those in previous frames)
                and "distance" (matching droplets to minimize center-to-center
                distances).
            refine (bool):
                Flag determining whether the droplet properties should be refined
                using fitting. This is a potentially slow procedure.
            num_processes (int or "auto"):
                Number of processes used for the refinement. If set to "auto", the
                number of processes is choosen automatically.
            progress (bool):
                Whether to show the progress of the process. If `None`, the progress is
                not shown, except for the first step if `refine` is `True`.

        Returns:
            :class:`DropletTrackList`: the resulting droplet tracks
        """
        etc = EmulsionTimeCourse.from_storage(
            storage, refine=refine, num_processes=num_processes, progress=progress
        )
        if progress is None:
            progress = False
        return cls.from_emulsion_time_course(etc, method=method, progress=progress)

    @classmethod
    def from_file(cls, path: str, *, progress: bool = True) -> DropletTrackList:
        """Create droplet track list by reading file.

        Args:
            path (str):
                The path from which the data is read. This function assumes that the
                data was written as an HDF5 file using :meth:`to_file`.
            progress (bool):
                Whether to show the progress of the process in a progress bar

        Returns:
            :class:`DropletTrackList`: an instance describing the droplet track list
        """
        import h5py

        obj = cls()
        with h5py.File(path, "r") as fp:
            # load the actual droplet track data and iterate in the right order
            for key in display_progress(
                sorted(fp.keys()), total=len(fp), enabled=progress
            ):
                dataset = fp[key]
                obj.append(DropletTrack._from_hdf_dataset(dataset))
        return obj

    def to_file(self, path: str, info: InfoDict | None = None) -> None:
        """Store data in hdf5 file.

        The data can be read using the classmethod :meth:`DropletTrackList.from_file`.

        Args:
            path (str):
                The path to which the data is written as an HDF5 file.
            info (dict):
                Additional data stored alongside the droplet track list
        """
        import h5py

        with h5py.File(path, "w") as fp:
            # write the actual emulsion data
            for i, droplet_track in enumerate(self):
                droplet_track._write_hdf_dataset(fp, f"track_{i:06d}")

            # write additional information
            if info:
                for k, v in info.items():
                    fp.attrs[k] = json.dumps(v)

    def remove_short_tracks(self, min_duration: float = 0) -> None:
        """Remove tracks that a shorter than a minimal duration.

        Args:
            min_duration (float):
                The minimal duration a droplet track must have in order to be retained.
                This is measured in actual time and not in the number of time steps
                stored in the track.
        """
        for i in reversed(range(len(self))):
            if self[i].duration <= min_duration:
                self.pop(i)

    @plot_on_axes()
    def plot(self, attribute: str = "radius", ax=None, **kwargs) -> PlotReference:
        """Plot the time evolution of all droplets.

        Args:
            attribute (str):
                The attribute to plot. Typical values include `radius` and
                `volume`, but others might be defined on the droplet class.
            {PLOT_ARGS}
            **kwargs:
                Additional keyword arguments are passed to the matplotlib plot
                function to affect the appearance. The special value `color="cycle"`
                implies that the default color cycle is used for the tracks, using
                different colors for different tracks.

        Returns:
            :class:`~pde.tools.plotting.PlotReference`: Information about the plot
        """
        # choose a suitable color for the tracks
        if "color" in kwargs:
            if kwargs["color"] == "cycle":
                kwargs.pop("color")  # if color is None, use the default color cycle
        else:
            kwargs["color"] = "k"  # use black by default

        # get maximal time
        if self:
            t_max = max(track.times[-1] for track in self if len(track.times) > 0)
        else:
            t_max = None

        # adjust alpha such that multiple tracks are visible well
        kwargs.setdefault("alpha", min(0.8, 20 / len(self)))
        elements = []
        for track in self:
            elements.append(
                track.plot(attribute=attribute, t_max=t_max, ax=ax, **kwargs).element
            )
            kwargs["label"] = ""  # set potential plot label only for first track

        return PlotReference(ax, elements, {"attribute": attribute})

    @plot_on_axes()
    def plot_positions(self, ax=None, **kwargs) -> PlotReference:
        """Plot all droplet tracks.

        Args:
            {PLOT_ARGS}
            **kwargs:
                Additional keyword arguments are passed to the matplotlib plot
                function to affect the appearance.

        Returns:
            :class:`~pde.tools.plotting.PlotReference`: Information about the plot
        """
        elements = [track.plot_positions(ax=ax, **kwargs).element for track in self]
        return PlotReference(ax, elements)


__all__ = ["DropletTrack", "DropletTrackList"]
