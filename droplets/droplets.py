"""
Classes representing (perturbed) droplets in various dimensions


.. autosummary::
   :nosignatures:

   SphericalDroplet
   DiffuseDroplet
   PerturbedDroplet2D
   PerturbedDroplet3D


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from abc import abstractmethod
import logging
from typing import (List, Optional, Callable, Dict, Any,  # @UnusedImport
                    TYPE_CHECKING, Sequence)  

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy import integrate

from pde.tools import spherical
from pde.grids.base import GridBase
from pde.fields import ScalarField
from pde.tools.misc import preserve_scalars
from pde.tools.cuboid import Cuboid



# work-around to satisfy type checking in python 3.6
if TYPE_CHECKING:
    # TYPE_CHECKING will always be False in production and this circular import
    # will thus be resolved.
    from .emulsions import EmulsionTimeCourse  # @UnusedImport



def get_dtype_field_size(dtype, field_name: str) -> int:
    """ return the number of elements in a field of structured numpy array
    
    Args:
        dtype: The dtype of the numpy array
        field_name (str): The name of the field that needs to be checked
    """
    shape = dtype.fields[field_name][0].shape
    return np.prod(shape) if shape else 1  # type: ignore
    
    
    
def iterate_in_pairs(it, fill=0):
    """ return consecutive pairs from an iterator
    
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



class DropletBase():
    """ represents a generic droplet """
    
    
    _subclasses: Dict[str, "DropletBase"] = {}  # collect all inheriting classes
 

    def __init__(self, **kwargs):
        r""" 
        Args:
            data (numpy.ndarray): A structured array carrying all information.
                If data is not supplied, the instance is initialized using the
                keyword arguments.
            \**kwargs: The keyword arguments are used to determine the data type
                using the classmethod :meth:`~get_datatype`.
        """
        # determine the data type of the droplet
        dtype = self.get_datatype(**kwargs)

        # create empty record with the correct data type
        self.data = np.recarray(1, dtype=dtype)[0]
            
        # data is stored in structured array with the given dtype. Note that
        # the conversion np.recarray(1)[0] turns the array into a scalar type
        # (instance of numpy.record) that contains the structured data.
        # This conversion is necessary for numba to operate on the data.
        

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.__name__] = cls


    def check_data(self):
        """ method that checks the validity and consistency of self.data """
        pass

        
    @classmethod
    def get_datatype(cls, **kwargs):
        """ default implementation to determine the type of the stored data """
        return [(key, 'f8') for key in kwargs.keys()]
        
    
    @property
    def _args(self):
        return {key: self.data[key] for key in self.data.dtype.names}

        
    def __str__(self):
        arg_list = [f'{key}={value}' for key, value in self._args.items()]
        return f"{self.__class__.__name__}({', '.join(arg_list)})"
    
    
    __repr__ = __str__
            
            
    @property
    def _data_array(self):
        """ return the data of the droplet in an unstructured array """
        return structured_to_unstructured(self.data)
            
            
    @classmethod
    def from_droplet(cls, droplet, **kwargs) -> "DropletBase": 
        r""" return a droplet with data taken from `droplet`
        
        Args:
            droplet: Another droplet from which data is copied
            \**kwargs: Additional arguments an be used to set data of the
                returned droplet.
        """
        args = droplet._args
        args.update(kwargs)
        return cls(**args)
    
    
    def copy(self, **kwargs) -> "DropletBase":
        r""" return a copy of the current droplet
        
        Args:
            \**kwargs: Additional arguments an be used to set data of the
                returned droplet.
        """
        return self.__class__.from_droplet(self, **kwargs)


    @property
    def data_bounds(self):
        """ lower and upper bounds on the parameters """
        num = len(self._data_array)
        return np.full(num, -np.inf), np.full(num, np.inf)
        
        
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return np.allclose(self._data_array, other._data_array,
                           rtol=0, atol=0, equal_nan=True)
        
        
    def __ne__(self, other):
        if self.__class__ != other.__class__:
            return True
        return not np.allclose(self._data_array, other._data_array,
                               rtol=0, atol=0, equal_nan=True)
        
        
        
class SphericalDroplet(DropletBase):
    """ Represents a single, spherical droplet
    
    Attributes:
        position (vector): center of the droplet
        radius (float): radius of the droplet
    """
    
        
    def __init__(self, position: Sequence[float], radius: float, **kwargs):
        r""" 
        Args:
            position (vector): center of the droplet
            radius (float): radius of the droplet
            \**kwargs: Additional arguments are forwarded to
                :meth:`get_datatype` to determine the correct data type.
        """
        position = np.atleast_1d(position)
        super().__init__(position=position, **kwargs)
        
        self.dim = get_dtype_field_size(self.data.dtype, 'position')
        self.position = position
        self.radius = radius
        self.check_data()


    def check_data(self):
        """ method that checks the validity and consistency of self.data """
        if self.radius < 0:
            raise ValueError('Radius must be positive')


    @classmethod
    def get_datatype(cls, **kwargs):
        dim = len(kwargs['position'])
        return [('position', 'f8', (dim,)), ('radius', 'f8')]


    @property
    def data_bounds(self):
        """ lower and upper bounds on the parameters """
        l, h = super().data_bounds
        l[self.dim] = 0  # radius must be non-negative
        return l, h
        
        
    @classmethod
    def from_volume(cls, position: Sequence[float], volume: float, **kwargs):
        """ Construct a droplet from given volume instead of radius
        
        Args:
            position (vector): center of the droplet
            volume (float): volume of the droplet
            interface_width (float, optional): width of the interface
        """
        dim = len(np.array(position, np.double, ndmin=1))
        radius = spherical.radius_from_volume(volume, dim)
        return cls(position, radius, **kwargs)


    @property
    def position(self):
        return self.data['position']
    
    @position.setter
    def position(self, value):
        value = np.asanyarray(value)
        if len(value) != self.dim:
            raise ValueError(f'Length of position must be {self.dim}')
        self.data['position'] = value


    @property
    def radius(self) -> float:
        return float(self.data['radius'])
    
    @radius.setter
    def radius(self, value: float):
        self.data['radius'] = value
        self.check_data()
        

    @property
    def volume(self) -> float:
        """ float: volume of the droplet """
        return spherical.volume_from_radius(self.radius, self.dim)
    
    @volume.setter
    def volume(self, volume: float):
        """ set the radius from a supplied volume """
        self.radius = spherical.radius_from_volume(volume, self.dim)
    

    @property
    def surface_area(self) -> float:
        """ float: surface area of the droplet """
        return spherical.surface_from_radius(self.radius, self.dim)


    @property
    def bbox(self) -> Cuboid:
        """ Cuboid: bounding box of the droplet """
        return Cuboid.from_points(self.position - self.radius,
                                  self.position + self.radius)
        
        
    def overlaps(self, other: "SphericalDroplet", grid: GridBase = None) \
            -> bool:
        """ determine whether another droplet overlaps with this one
        
        Note that this function so far only compares the distances of the
        droplets to their radii, which does not respect perturbed droplets
        correctly.        
        
        Args:
            other (SphericalDroplet): instance of the other droplet
            grid (GridBase): grid that determines how distances are measured,
                which is for instance important to respect periodic boundary
                conditions. If omitted, an Eucledian distance is assumed.
                
        Returns:
            bool: whether the droplets overlap or not
        """
        if grid is None:
            distance = np.linalg.norm(self.position - other.position)
        else:
            distance = grid.distance_real(self.position, other.position)
        return distance < self.radius + other.radius  # type: ignore
    
    
    @preserve_scalars
    def interface_position(self, φ):
        """ calculates the position of the interface of the droplet
        
        Args:
            φ (float or array): The angle in the polar coordinate system that
                is used to describe the interface

        Returns:
            An array with the coordinates of the interfacial points associated
            with each angle given by `φ`.
            
        Raises:
            ValueError: If the dimension of the space is not 2
        """
        if self.dim != 2:
            raise ValueError('Interfacial position only supported for 2d '
                             'grids')
        pos = self.radius * np.transpose([np.cos(φ), np.sin(φ)])
        return self.position[None, :] + pos
    
    
    @property
    def interface_curvature(self):
        """ float: the mean curvature of the interface of the droplet
        """
        return 1 / self.radius

        
    def _get_phase_field(self, grid: GridBase, dtype=np.double):
        """ Creates an image of the droplet on the `grid`
        
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field
            dtype:
                The numpy data type defining the type of the returned image.
                If `dtype == np.bool`, a binary representation is returned.
                
        Returns:
            An array with data values representing the droplet phase field at
            support points of the `grid`.
        """
        if self.dim != grid.dim:
            raise ValueError(f'Droplet dimension ({self.dim}) incompatible '
                             f'with grid dimension ({grid.dim})')
        
        # calculate distances from droplet center
        dist = grid.polar_coordinates_real(self.position)
        return (dist < self.radius).astype(dtype)

        
    def get_phase_field(self, grid: GridBase,
                        label: str = None,
                        dtype=np.double) -> ScalarField:
        """ Creates an image of the droplet on the `grid`
        
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field
            label (str):
                The label associated with the returned scalar field
            dtype:
                The numpy data type of the returned array. Typical choices are
                `numpy.double` and `numpy.bool`.
        
        Returns:
            :class:`~pde.fields.ScalarField`: A scalar field
            representing the droplet
        """
        data = self._get_phase_field(grid, dtype=dtype)
        return ScalarField(grid, data=data, label=label)
        
        
    def get_phase_field_masked(self, grid: GridBase, mask):
        """ Returns an image of the droplet restricted to the given `mask`
        
        Args: 
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field
            mask (array):
                A binary mask marking the support points whose data is returned
                
        Returns:
            An array with the values at the support points marked in `mask`
        """
        return self._get_phase_field(grid)[mask]
    
        
    def plot_on_axes(self, ax, **kwargs):
        """ Plot the droplet
        
        Args:
            ax (:class:`matplotlib.axes.Axes`):
                The axes to which the droplet picture is added
            **kwargs:
                Additional keyword arguments are passed to the matplotlib Circle
                class to affect the appearance.
        """
        import matplotlib.pyplot as plt
        
        if self.dim != 2:
            raise NotImplementedError('Plotting is only implemented in 2d')

        kwargs.setdefault('fill', False)
        circle = plt.Circle(self.position, self.radius, **kwargs)
        ax.add_artist(circle) 
        


class DiffuseDroplet(SphericalDroplet):
    """ Represents a single, spherical droplet with a diffuse interface
    
    Attributes:
        position (vector): center of the droplet
        radius (float): radius of the droplet
        interface_width (float, optional): width of the interface
    """
    
        
    def __init__(self, position: Sequence[float],
                 radius: float,
                 interface_width: float = None, **kwargs):
        r""" 
        Args:
            position (vector): center of the droplet
            radius (float): radius of the droplet
            interface_width (float, optional): width of the interface
            \**kwargs: Additional arguments are forwarded to
                :meth:`get_datatype` to determine the correct data type.
        """
        super().__init__(position=position, radius=radius, **kwargs)
        self.interface_width = interface_width


    @property
    def data_bounds(self):
        """ lower and upper bounds on the parameters """
        l, h = super().data_bounds
        l[self.dim + 1] = 0  # interface width must be non-negative
        return l, h


    @classmethod
    def get_datatype(cls, **kwargs):
        return super().get_datatype(**kwargs) + [('interface_width', 'f8')]
        

    @property
    def interface_width(self) -> Optional[float]:
        if np.isnan(self.data['interface_width']):
            return None
        else:
            return float(self.data['interface_width'])       
    
    @interface_width.setter
    def interface_width(self, value: Optional[float]):
        if value is None:
            self.data['interface_width'] = np.nan
        elif value < 0:
            raise ValueError('Interface width must not be negative')
        else:
            self.data['interface_width'] = value
        self.check_data()

        
    def _get_phase_field(self, grid: GridBase, dtype=np.double):
        """ Creates an image of the droplet on the `grid`
        
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid used for discretizing the droplet phase field
            dtype:
                The numpy data type defining the type of the returned image.
                If `dtype == np.bool`, a binary representation is returned.
                
        Returns:
            An array with data values representing the droplet phase field at
            support points of the `grid`.
        """
        if self.dim != grid.dim:
            raise ValueError(f'Droplet dimension ({self.dim}) incompatible '
                             f'with grid dimension ({grid.dim})')
        
        if self.interface_width is None:
            interface_width = grid.typical_discretization
        else:
            interface_width = self.interface_width

        # calculate distances from droplet center
        dist = grid.polar_coordinates_real(self.position)

        # make the image
        if interface_width == 0 or dtype == np.bool:
            result = (dist < self.radius)
        else:
            result = 0.5 + 0.5 * np.tanh((self.radius - dist) / interface_width)

        return result.astype(dtype)

        

class _PerturbedDropletND(DiffuseDroplet):
    """ represents a single droplet with a perturbed shape.
    
    This acts as an abstract class for which member functions need to specified
    depending on dimensionality. 
    """
    

    def __init__(self, position: List[float],
                 radius: float,
                 interface_width: float = None,
                 amplitudes: List[float] = None):
        """ initialize perturbed droplet """
        super().__init__(position=position, radius=radius,
                         interface_width=interface_width, amplitudes=amplitudes)
        self.amplitudes = amplitudes

        if len(self.position) != self.__class__.dim:
            raise ValueError(f'Space dimension must be {self.__class__.dim}')


    @classmethod
    def get_datatype(cls, **kwargs):
        amplitudes = kwargs.get('amplitudes')
        if amplitudes is None:
            modes = 0
        else:
            modes = len(amplitudes)
        return super().get_datatype(**kwargs) + [('amplitudes', 'f8', (modes,))]


    @property
    def data_bounds(self):
        """ lower and upper bounds on the parameters """
        l, h = super().data_bounds
        n = self.dim + 2
        # relative perturbation amplitudes must be between [-1, 1]
        l[n:n + self.modes] = -1
        h[n:n + self.modes] = 1  
        return l, h

    
    @property
    def modes(self) -> int:
        """ int: number of perturbation modes """
        shape = self.data.dtype.fields['amplitudes'][0].shape
        return int(shape[0]) if shape else 1
    
        
    @property
    def amplitudes(self):
        return np.atleast_1d(self.data['amplitudes'])
    
    @amplitudes.setter
    def amplitudes(self, value: float):
        self.data['amplitudes'] = np.broadcast_to(value, (self.modes,))
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

        
    def _get_phase_field(self, grid: GridBase, dtype=np.double):
        """ create an image of the droplet on the `grid`
        
        `dtype` defines the type of the returned image. If dtype == np.bool, a
            binary representation is returned. 
        """
        if self.dim != grid.dim:
            raise ValueError(f'Droplet dimension ({self.dim}) incompatible '
                             f'with grid dimension ({grid.dim})')
        
        if self.interface_width is None:
            interface_width = grid.typical_discretization
        else:
            interface_width = self.interface_width

        # calculate grid distance from droplet center
        dist, *angles = grid.polar_coordinates_real(self.position,
                                                    ret_angle=True)

        # calculate interface distance from droplet center
        interface = self.interface_distance(*angles)
            
        # make the image
        if interface_width == 0 or dtype == np.bool:
            result = (dist < interface)
        else:
            result = 0.5 + 0.5*np.tanh((interface - dist) / interface_width)

        return result.astype(dtype)

        
    def get_phase_field(self, grid: GridBase, label: str = None,
                        dtype=np.double) -> ScalarField:
        """ create an image of the droplet on the `grid`
        
        `dtype` defines the type of the returned image. If dtype == np.bool, a
            binary representation is returned.
        `label` sets an identifier for the phase field 
        """
        return ScalarField(grid, data=self._get_phase_field(grid, dtype=dtype),
                           label=label)
        
        
    def get_phase_field_masked(self, grid: GridBase, mask):
        """ create an image of the droplet on the `grid` and return only the
        points that are non-zero in the mask """
        return self._get_phase_field(grid)[mask]
    


class PerturbedDroplet2D(_PerturbedDropletND):
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
    
    Attributes:
        position (vector): droplet center position
        radius (float): base radius :math:`R_0` of the droplet
        interface_width (float, optional): width of the interface
        amplitudes (array, optional): (dimensionless) perturbation amplitudes
            :math:`\{\epsilon^{(1)}_1, \epsilon^{(2)}_1, 
            \epsilon^{(1)}_2, \epsilon^{(2)}_2,
            \epsilon^{(1)}_3, \epsilon^{(2)}_3, \dots \}`
    """
    
    dim = 2
        
                
    def __init__(self, position: List[float],
                 radius: float,
                 interface_width: float = None,
                 amplitudes: List[float] = None):
        """ initialize perturbed droplet """
        super().__init__(position, radius, interface_width, amplitudes)
        if len(self.amplitudes) % 2 != 0:
            logger = logging.getLogger(self.__class__.__module__)
            logger.warning('`amplitudes` should be of even length to capture '
                           'all perturbations of the highest mode.')
        
        
    @preserve_scalars
    def interface_distance(self, φ):
        """ calculates the distance of the droplet interface to the origin
        
        Args:
            φ (float or array): The angle in the polar coordinate system that
                is used to describe the interface

        Returns:
            An array with the distances of the interfacial points associated
            with each angle given by `φ`.
        """
        dist = np.ones(φ.shape, dtype=np.double)
        for n, (a, b) in enumerate(iterate_in_pairs(self.amplitudes), 1):
            if a != 0:
                dist += a * np.sin(n * φ)
            if b != 0:
                dist += b * np.cos(n * φ)
        return self.radius * dist
        
        
    @preserve_scalars
    def interface_position(self, φ):
        """ calculates the position of the interface of the droplet
        
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
        r""" calculates the mean curvature of the interface of the droplet
        
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
        for n, (a, b) in enumerate(iterate_in_pairs(self.amplitudes), 1):
            factor = n*n - 1
            if a != 0:
                curv_radius -= a * factor * np.sin(n * φ)
            if b != 0:
                curv_radius -= b * factor * np.cos(n * φ)
        return 1 / (self.radius * curv_radius)
    

    @property
    def volume(self) -> float:
        """ float: volume of the droplet """
        term = 1 + np.sum(self.amplitudes**2) / 2
        return np.pi * self.radius**2 * term  # type: ignore

    @volume.setter
    def volume(self, volume: float):
        """ set volume keeping relative perturbations """
        term = 1 + np.sum(self.amplitudes**2) / 2
        self.radius = np.sqrt(volume / (np.pi * term))
        
        
    @property
    def surface_area(self) -> float:
        """ float: surface area of the droplet """
        # discretize surface for simple approximation to integral
        φs, dφ = np.linspace(0, 2*np.pi, 256, endpoint=False, retstep=True)
        
        dist = np.ones(φs.shape, dtype=np.double)
        dist_dφ = np.zeros(φs.shape, dtype=np.double)
        for n, (a, b) in enumerate(iterate_in_pairs(self.amplitudes), 1):
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
        for n, (a, b) in enumerate(iterate_in_pairs(self.amplitudes), 1):
            length += n**2 * (a**2 + b**2)
        return np.pi * self.radius * length / 2  # type: ignore
        
        
    def plot_on_axes(self, ax, **kwargs):
        """ Plot the perturbed droplet
        
        Args:
            ax (:class:`matplotlib.axes.Axes`):
                The axes to which the droplet picture is added
            **kwargs:
                Additional keyword arguments are passed to the matplotlib plot
                function to affect the appearance.
        """
        if self.dim != 2:
            raise NotImplementedError('Plotting is only implemented in 2d')

        φ = np.linspace(0, 2*np.pi)
        xy = self.interface_position(φ)
        kwargs.setdefault('color', 'k')
        ax.plot(xy[:, 0], xy[:, 1], **kwargs)
        
        

class PerturbedDroplet3D(_PerturbedDropletND):
    r""" Represents a single droplet in two dimensions with a perturbed shape
    
    The shape is described using the distance :math:`R(\theta, \phi)` of the
    interface from the origin as a function of the azimuthal angle
    :math:`\theta` and the polar angle :math:`\phi`. This function is developed
    as a truncated series of spherical harmonics :math:`Y_{l,m}(\theta, \phi)`:
    
    .. math::
        R(\phi) = R_0 \left[1 + \sum_{l=1}^{N_l}\sum_{m=-l}^l
                                \epsilon_{l,m} Y_{l,m}(\theta, \phi) \right]
                            
    where :math:`N_l` is the number of perturbation modes considered, which is 
    deduced from the length of the `amplitudes` array.
    
    Attributes:
        position (vector): droplet center, i.e., coordinates of the origin
        radius (float): base radius :math:`R_0` of the droplet
        interface_width (float, optional): width of the interface
        amplitudes (array, optional): perturbation amplitudes
            :math:`\epsilon_{l,m}`
            
    """
    
    dim = 3
        
                
    def __init__(self, position: List[float],
                 radius: float,
                 interface_width: float = None,
                 amplitudes: List[float] = None):
        """ initialize perturbed droplet """
        super().__init__(position, radius, interface_width, amplitudes)
        num_modes = len(self.amplitudes) + 1
        if not spherical.spherical_index_count_optimal(num_modes):
            logger = logging.getLogger(self.__class__.__module__)
            l, _ = spherical.spherical_index_lm(num_modes)
            opt_modes = spherical.spherical_index_count(l) - 1
            logger.warning('The length of `amplitudes` should be such that all '
                           'orders are captured for the perturbations with the '
                           f'highest degree ({l}). Consider increasing the '
                           f'size of the array to {opt_modes}.')
        
        
    @preserve_scalars
    def interface_distance(self, θ, φ):
        r""" calculates the distance of the droplet interface to the origin
        
        Args:
            θ (float or array): Azimuthal angle (in :math:`[0, \pi]`)
            φ (float or array): Polar angle (in :math:`[0, 2\pi]`)

        Returns:
            An array with the distances of the interfacial points associated
            with the angles.
        """
        assert θ.shape == φ.shape
        dist = np.ones(φ.shape, dtype=np.double)
        for k, a in enumerate(self.amplitudes, 1):
            if a != 0:
                dist += a * spherical.spherical_harmonic_real_k(k, θ, φ)
        return self.radius * dist
        
        
    @preserve_scalars
    def interface_position(self, θ, φ):
        r""" calculates the position of the interface of the droplet
        
        Args:
            θ (float or array): Azimuthal angle (in :math:`[0, \pi]`)
            φ (float or array): Polar angle (in :math:`[0, 2\pi]`)

        Returns:
            An array with the coordinates of the interfacial points associated
            with the angles.
        """
        dist = self.interface_distance(θ, φ)
        unit_vector = [np.sin(θ) * np.cos(φ),
                       np.sin(θ) * np.sin(φ),
                       np.cos(θ)]
        pos = dist[:, None] * np.transpose(unit_vector)
        return self.position[None, :] + pos

        
    @preserve_scalars
    def interface_curvature(self, θ, φ):
        r""" calculates the mean curvature of the interface of the droplet
        
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
        for k, a in enumerate(self.amplitudes, 1):
            if a != 0:
                l, _ = spherical.spherical_index_lm(k)
                hk = (l**2 + l - 2) / 2
                correction = a * hk * Yk(k, θ, φ)
        return 1 / self.radius + correction / self.radius**2
        
        
    @property
    def volume(self) -> float:
        """ float: volume of the droplet (determined numerically) """
        def integrand(θ, φ):
            """ helper function calculating the integrand """
            r = self.interface_distance(θ, φ)
            return r**3 * np.sin(θ) / 3
        
        volume = integrate.dblquad(integrand, 0, 2 * np.pi,
                                   lambda _: 0, lambda _: np.pi)[0]
        return volume  # type: ignore
    

    @volume.setter
    def volume(self, volume: float):
        """ set volume keeping relative perturbations """
        raise NotImplementedError('Cannot set volume')
                    
                    
    @property
    def volume_approx(self) -> float:            
        """ float: approximate volume to linear order in the perturbation """
        volume = spherical.volume_from_radius(self.radius, 3)
        if len(self.amplitudes) > 0:
            volume += self.amplitudes[0] * 2 * np.sqrt(np.pi) * self.radius**2
        return volume



def droplet_from_data(droplet_class: str, data) -> DropletBase:
    """ create a droplet instance of the given class using some data
    
    Args:
        droplet_class (str): The name of the class that is used to create the
            droplet instance
        data (numpy.ndarray): A numpy array that defines the droplet properties
    """
    cls = DropletBase._subclasses[droplet_class]
    return cls(**{key: data[key] for key in data.dtype.names})  # type: ignore
        
        
        
__all__ = ["SphericalDroplet", "DiffuseDroplet", "PerturbedDroplet2D",
           "PerturbedDroplet3D"]
