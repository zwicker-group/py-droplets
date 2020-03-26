"""
Classes representing the time evolution of droplets

.. autosummary::
   :nosignatures:

   DropletTrack
   DropletTrackList


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json
from typing import (List, Optional, Callable, Dict, Any,  # @UnusedImport
                    Union, TYPE_CHECKING)

import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.ndimage import filters

from pde.grids.base import GridBase
from pde.trackers.base import InfoDict

from .droplets import SphericalDroplet, droplet_from_data

if TYPE_CHECKING:
    from .emulsions import EmulsionTimeCourse  # @UnusedImport



def contiguous_true_regions(condition: np.ndarray) -> np.ndarray:
    """ Finds contiguous True regions in the boolean array "condition"
    
    Inspired by http://stackoverflow.com/a/4495197/932593
    
    Args:
        condition (:class:`numpy.ndarray`):
            A one-dimensional boolean array
            
    
    Returns:
        :class:`numpy.ndarray`: A two-dimensional array where the first column
        is the start index of the region and the second column is the end index
    """
    if len(condition) == 0:
        return []
    
    # convert condition array to integer
    condition = np.asarray(condition, np.int)
    
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
    return idx.reshape(-1, 2)


            
class DropletTrack():
    """ information about a single droplet over multiple time steps """
    
    def __init__(self, droplets=None, times=None):
        """
        Args:
            emulsions (list): List of emulsions that describe this time course
            times (list): Times associated with the emulsions
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
            raise ValueError('The list of droplets and the list of times need '
                             'to have the same length')
        
        
    def __repr__(self):
        """ human-readable representation of a droplet track """
        class_name = self.__class__.__name__
        if len(self.times) == 0:
            return f'{class_name}([])'
        elif len(self.times) == 1:
            return f'{class_name}(time={self.start})'
        else:
            return f'{class_name}(timespan={self.start}..{self.end})'
        
        
    def __len__(self):
        """ number of time points """
        return len(self.times)
    
    
    def __getitem__(self, key: Union[int, slice]):
        """ return the droplets identified by the given index/slice """
        result = self.droplets.__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(droplets=result, times=self.times[key])
        else:
            return result


    def __eq__(self, other):
        """ determine whether two DropletTracks instance are equal """
        return (self.times == other.times and
                self.droplets == other.droplets) 

    
    @property
    def start(self) -> float:
        """ float: first time point """
        return self.times[0]  # type: ignore

    @property
    def end(self) -> float:
        """ float: last time point """
        return self.times[-1]  # type: ignore
    
    @property
    def duration(self) -> float:
        """ float: total duration of the track """
        if len(self.times) > 0:
            return self.end - self.start
        else:
            return 0
    
    @property
    def first(self) -> SphericalDroplet:
        """ SphericalDroplet: first droplet instance """
        return self.droplets[0]  # type: ignore

    @property
    def last(self) -> SphericalDroplet:
        """ SphericalDroplet: last droplet instance """
        return self.droplets[-1]  # type: ignore
    
    @property
    def dim(self) -> Optional[int]:
        """ return the space dimension of the droplets """
        try:
            return self.last.dim
        except IndexError:
            return None
        
        
    @property
    def data(self):
        """ numpy.ndarray: an array containing the data of the full track """
        if len(self) == 0:
            return None
        else:
            d0 = self.first
            dtype = [('time', 'f8')] + d0.data.dtype.descr
            result = np.empty(len(self), dtype=dtype)
            for i in range(len(self)):
                result[i] = (self.times[i],) + self.droplets[i].data.tolist()
            return result
    
    
    def __iter__(self):
        """ iterate over all droplets """
        return iter(self.droplets)
    
    
    def items(self):
        """ iterate over all times and droplets, returning them in pairs """
        return zip(self.times, self.droplets)        


    def append(self, droplet: SphericalDroplet,
               time: Optional[float] = None):
        """ append a new droplet with a time code
        
        Args:
            droplet (SphericalDroplet): the droplet
            time (float, optional): The time point
        """
        if self.dim is not None and droplet.dim != self.dim:
            raise ValueError("Space dimension of the DropletTrack is unequal "
                             "to that of the added droplet")
    
        # add the emulsion
        self.droplets.append(droplet.copy())
        if time is None:
            time = 0 if len(self.times) == 0 else self.times[-1] + 1
        self.times.append(time)
    
    
    def get_position(self, time: float):
        """ numpy.ndarray: returns the droplet position at a specific time """
        try:
            idx = self.times.index(time)
        except AttributeError:
            # assume that self.times is a numpy array
            idx = np.nonzero(self.times == time)[0][0]
        return self.droplets[idx].position
        
    
    def get_trajectory(self, smoothing: float = 0):
        """ return a list of positions over time
        
        Args:
            smoothing (float): Determines the length scale for some gaussian
                smoothing of the trajectory. Setting this to zero disables
                smoothing.
        
        Returns:
            numpy.ndarray: An array giving the position of the droplet at each
                time instance
        """
        trajectory = np.array([droplet.position for droplet in self.droplets])
        if smoothing:
            filters.gaussian_filter1d(trajectory, output=trajectory,
                                      sigma=smoothing, axis=0, mode='nearest')
        return trajectory
        
        
    def get_radii(self) -> np.ndarray:
        """ returns the droplet radius for each time point """
        return np.array([droplet.radius for droplet in self.droplets])
        
    
    def get_volumes(self) -> np.ndarray:
        """ returns the droplet volume for each time point """
        return np.array([droplet.volume for droplet in self.droplets])
        
    
    def time_overlaps(self, other: "DropletTrack") -> bool:
        """ determine whether two DropletTrack instances overlaps in time
        
        Args:
            other (DropletTrack): The other droplet track
            
        Returns:
            bool: True when both tracks contain droplets at the same time step
        """
        s0, s1 = self.start, self.end
        o0, o1 = other.start, other.end
        return (s0 <= o1 and o0 <= s1)
    
    
    @classmethod
    def _from_dataset(cls, dataset) -> "DropletTrack":
        """ construct a droplet track by reading data from an hdf5 dataset
        
        Args:
            dataset:
                an HDF5 dataset from which the data of the droplet track is read
        """
        # there are values, so the emulsion is not empty
        droplet_class = dataset.attrs['droplet_class']
        obj = cls() 
        if droplet_class == 'None':
            return obj
        else:
            # separate time from the data set
            times = dataset['time']
            droplet_data = rfn.rec_drop_fields(dataset, 'time')
            for time, data in zip(times, droplet_data):
                droplet = droplet_from_data(droplet_class, data)
                obj.append(droplet, time=time)  # type: ignore

        return obj


    @classmethod
    def from_file(cls, filename: str) -> "DropletTrack":
        """ create droplet track by reading from file
        
        Args:
            filename (str): Name of the file to read emulsion from
        """
        import h5py
        with h5py.File(filename, "r") as fp:
            if len(fp) != 1:
                raise RuntimeError('Multiple droplet track were found in the '
                                   f'file {filename})') 
            dataset = fp[list(fp.keys())[0]]  # retrieve the only dataset
            obj = cls._from_dataset(dataset)
            
        return obj


    def _write_hdf_dataset(self, fp, key: str = 'droplet_track'):
        """ write data to a given hdf5 file pointer `fp` """
        if self:
            # emulsion contains at least one droplet
            dataset = fp.create_dataset(key, data=self.data)
            # self.data ensures that there is only one droplet class
            dataset.attrs['droplet_class'] = self[0].__class__.__name__
            
        else:
            # create empty dataset to indicate empty emulsion
            dataset = fp.create_dataset(key, shape=tuple())
            dataset.attrs['droplet_class'] = 'None'
            
        return dataset
        
            
    def to_file(self, filename: str, info: InfoDict = None):
        """ store data in hdf5 file

        Args:
            filename (str): Name of the file to write emulsion to
        """
        import h5py
        with h5py.File(filename, "w") as fp:
            self._write_hdf_dataset(fp)
            
            # write additional information
            if info:
                for k, v in info.items():
                    fp.attrs[k] = json.dumps(v)
                
    
    def plot(self, attribute: str = 'radius', **kwargs):
        """ plot the time evolution of the droplet
        
        Args:
            attribute (str):
                The attribute to plot. Typical values include `radius` and
                `volume`, but others might be defined on the droplet class.
            **kwargs:
                Additional keyword arguments are passed to the matplotlib plot
                function to affect the appearance.
        """
        import matplotlib.pyplot as plt

        if len(self.times) == 0:
            return

        if attribute in {'radius', 'radii'}:
            data = self.get_radii()
        elif attribute in {'volume', 'volumes'}:
            data = self.get_volumes()
        else:
            data = [getattr(droplet, attribute) for droplet in self.droplets]

        plt.plot(self.times, data, **kwargs)
        plt.xlabel('Time')
        plt.ylabel('Radius')

    
    def plot_positions(self, grid: Optional[GridBase] = None,
                       arrow: bool = True, **kwargs):
        """ plot the droplet track
        
        Args:
            grid (GridBase, optional): The grid on which the droplets are
                defined. If given, periodic boundary conditions can be respected
                in the plotting.
            arrow (bool, optional): Flag determining whether an arrow head is
                shown to indicate the direction of the droplet drift.
            **kwargs:
                Additional keyword arguments are passed to the matplotlib plot
                function to affect the appearance.
        """
        import matplotlib.pyplot as plt

        if len(self.times) == 0:
            return

        if self.dim != 2:
            raise NotImplementedError('Plotting is only implemented for '
                                      'two-dimensional spaces.')
        
        # obtain droplet positions as a function of time
        xy = self.get_trajectory()
        
        if grid is None:
            # simply plot the trajectory
            line, = plt.plot(xy[:, 0], xy[:, 1], **kwargs)
            
        else:
            # use the grid to detect wrapping around
            segments = []
            for p1, p2 in zip(xy[:-1], xy[1:]):
                dist_direct = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                dist_real = grid.distance_real(p1, p2)
                close = np.isclose(dist_direct, dist_real)
                segments.append(close)            
                
            # plot the individual segments
            line = None
            for s, e in contiguous_true_regions(segments):
                if line is None:
                    color = kwargs.get('color')
                else:
                    color = line.get_color()  # ensure colors stays the same
                line, = plt.plot(xy[s:e + 1, 0], xy[s:e + 1, 1], color=color)
        
        if arrow:
            size = min(sum(plt.xlim()), sum(plt.ylim()))
            dx = xy[-1] - xy[-2]
            plt.arrow(xy[-2, 0], xy[-2, 1], dx[0], dx[1],
                      head_width=0.02 * size, color=line.get_color())
        
        
        
class DropletTrackList(list):
    """ a list of instances of :class:`DropletTrack` """

    
    @classmethod
    def from_emulsion_time_course(cls, time_course: "EmulsionTimeCourse"):
        """ create the instance from an emulsion time course """
        # get tracks, i.e. clearly overlapping droplets
        tracks = cls()
        t_last = None
        for t, emulsion in time_course.items():
            # determine live tracks
            tracks_alive = [track for track in tracks if track.end <= t_last]

            # handle all droplets in the emulsion
            for droplet in emulsion:
                overlap_count = 0
                overlap_track: Optional[DropletTrack] = None
                # iterate over all active tracks
                for track in tracks_alive:
                    if track.last.overlaps(droplet, time_course.grid):
                        overlap_count += 1
                        overlap_track = track

                if overlap_count == 1:
                    overlap_track.append(droplet, time=t)  # type: ignore
                else:
                    tracks.append(DropletTrack(droplets=[droplet], times=[t]))

            t_last = t
        return tracks


    def __getitem__(self, key: Union[int, slice]):
        """ return the droplets identified by the given index/slice """
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(result)
        else:
            return result


    def remove_short_tracks(self, min_duration: float = 0):
        """ remove tracks that a shorter than a minimal duration
        
        Args:
            min_duration (float): The minimal duration a droplet track must have
                in order to be retained. This is measured in actual time and not
                in the number of time steps stored in the track
        """
        for i in reversed(range(len(self))):
            if self[i].duration <= min_duration:
                self.pop(i)
        
        
    @classmethod
    def from_file(cls, filename: str) -> "DropletTrackList":
        """ create droplet track list by reading file
        
        Args:
            filename (str): The filename from which the data is read
            
        Returns:
            DropletTrackList: an instance describing the droplet track list
        """
        import h5py
        obj = cls()
        with h5py.File(filename, "r") as fp:
            for key in sorted(fp.keys()):  # iterate in the stored order
                dataset = fp[key]
                obj.append(DropletTrack._from_dataset(dataset))
        return obj
    
                
    def to_file(self, filename: str, info: InfoDict = None):
        """ store data in hdf5 file
        
        Args:
            filename (str): determines the location where the file is written
            info (dict): can be additional data stored alongside 
        """
        import h5py
        with h5py.File(filename, "w") as fp:
            # write the actual emulsion data
            for i, droplet_track in enumerate(self):
                droplet_track._write_hdf_dataset(fp, f"track_{i:06d}")
                
            # write additional information
            if info:
                for k, v in info.items():
                    fp.attrs[k] = json.dumps(v)
                
                        
    def plot(self, attribute: str = 'radius', **kwargs):
        """ plot the time evolution of all droplets
        
        Args:
            attribute (str):
                The attribute to plot. Typical values include `radius` and
                `volume`, but others might be defined on the droplet class.
            **kwargs:
                Additional keyword arguments are passed to the matplotlib plot
                function to affect the appearance.
        """
        kwargs.setdefault('color', 'k')
        # adjust alpha such that multiple tracks are visible well
        kwargs.setdefault('alpha', np.exp(-(len(self) - 1) / 30))
        for track in self:
            track.plot(attribute=attribute, **kwargs)
                        
                        
    def plot_positions(self, **kwargs):
        """ plot all droplet tracks
        
        Args:
            **kwargs:
                Additional keyword arguments are passed to the matplotlib plot
                function to affect the appearance.
        """
        for track in self:
            track.plot_positions(**kwargs)
        
        
        
__all__ = ["DropletTrack", "DropletTrackList"]
