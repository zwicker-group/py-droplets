"""
Functions for analyzing phase field images of emulsions.

.. autosummary::
   :nosignatures:

   locate_droplets
   refine_droplet
   get_structure_factor
   get_length_scale


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
import math
import warnings
from functools import reduce
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)
from scipy import ndimage, optimize

try:
    from pyfftw.interfaces.numpy_fft import fftn as np_fftn
except ImportError:
    from numpy.fft import fftn as np_fftn

from pde.fields import ScalarField
from pde.grids import CylindricalSymGrid
from pde.grids.base import GridBase
from pde.grids.cartesian import CartesianGrid
from pde.grids.spherical import SphericalSymGridBase
from pde.tools.math import SmoothData1D
from pde.tools.typing import NumberOrArray

from .droplets import (
    DiffuseDroplet,
    PerturbedDroplet2D,
    PerturbedDroplet3D,
    SphericalDroplet,
)
from .emulsions import Emulsion


def threshold_otsu(data: np.ndarray, nbins: int = 256) -> float:
    """Find the threshold value for a bimodal histogram using the Otsu method.

    If you have a distribution that is bimodal, i.e., with two peaks and a valley
    between them, then you can use this to find the location of that valley, which
    splits the distribution into two.

    Args:
        data (:class:`~numpy.ndarray`):
            The data to be analyzed
        nbins (int):
            The number of bins in the histogram, which defines the accuracy of the
            determined threshold.

    Modified from https://stackoverflow.com/a/71345917/932593, which is based on the
    the SciKit Image threshold_otsu implementation:
    https://github.com/scikit-image/scikit-image/blob/70fa904eee9ef370c824427798302551df57afa1/skimage/filters/thresholding.py#L312
    """
    counts, bin_edges = np.histogram(data.flat, bins=nbins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    return float(bin_centers[idx])


def _locate_droplets_in_mask_cartesian(
    grid: CartesianGrid, mask: np.ndarray
) -> Emulsion:
    """locate droplets in a (potentially periodic) data set on a Cartesian grid

    This function locates droplets respecting periodic boundary conditions.

    Args:
        mask (:class:`~numpy.ndarray`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    if mask.shape != grid.shape:
        raise ValueError(
            f"The shape {mask.shape} of the data is not compatible with the grid "
            f"shape {grid.shape}"
        )

    # pad the array to simulate periodic boundary conditions
    offset = np.array([dim if p else 0 for p, dim in zip(grid.periodic, grid.shape)])
    pad = np.c_[offset, offset].astype(np.intc)
    mask_padded = np.pad(mask, pad, mode="wrap")
    assert np.all(mask_padded.shape == np.array(grid.shape) + 2 * offset)

    # locate individual clusters in the padded image
    labels, num_labels = ndimage.label(mask_padded)
    if num_labels == 0:
        return Emulsion([], grid=grid)
    indices = range(1, num_labels + 1)

    # create and emulsion from this of droplets
    grid._logger.info(f"Found {num_labels} droplet candidate(s)")

    # determine position from binary image and scale it to real space
    positions = ndimage.center_of_mass(mask_padded, labels, index=indices)
    # correct for the additional padding of the array
    positions = grid.transform(positions - offset, "cell", "grid")

    # determine volume from binary image and scale it to real space
    volumes = ndimage.sum(mask_padded, labels, index=indices)
    volumes = np.asanyarray(volumes) * np.prod(grid.discretization)

    # only retain droplets that are inside the central area
    droplets = (
        SphericalDroplet.from_volume(position, volume)
        for position, volume in zip(positions, volumes)
        if grid.cuboid.contains_point(position)
    )

    # filter overlapping droplets (e.g. due to duplicates)
    emulsion = Emulsion(droplets, grid=grid)
    num_candidates = len(emulsion)
    if num_candidates < num_labels:
        grid._logger.info(f"Only {num_candidates} candidate(s) inside bounds")

    emulsion.remove_overlapping()
    if len(emulsion) < num_candidates:
        grid._logger.info(f"Only {num_candidates} candidate(s) not overlapping")

    return emulsion


def _locate_droplets_in_mask_spherical(
    grid: SphericalSymGridBase, mask: np.ndarray
) -> Emulsion:
    """locates droplets in a binary data set on a spherical grid

    Args:
        mask (:class:`~numpy.ndarray`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    assert np.all(mask.shape == grid.shape)

    # locate clusters in the binary image
    labels, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return Emulsion([], grid=grid)

    # locate clusters around origin
    object_slices = ndimage.find_objects(labels)
    droplet = None
    for slices in object_slices:
        if slices[0].start == 0:  # contains point around origin
            radius = float(grid.transform(slices[0].stop, "cell", "grid").flat[-1])
            droplet = SphericalDroplet(np.zeros(grid.dim), radius=radius)
        else:
            logger = logging.getLogger(grid.__class__.__module__)
            logger.warning("Found object not located at origin")

    # return an emulsion of droplets
    if droplet:
        return Emulsion([droplet], grid=grid)
    else:
        return Emulsion([], grid=grid)


def _locate_droplets_in_mask_cylindrical_single(
    grid: CylindricalSymGrid, mask: np.ndarray
) -> Emulsion:
    """locate droplets in a data set on a single cylindrical grid

    Args:
        mask (:class:`~numpy.ndarray`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    # locate the individual clusters
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return Emulsion([], grid=grid)

    # locate clusters on the symmetry axis
    object_slices = ndimage.find_objects(labels)
    indices = []
    for index, slices in enumerate(object_slices, 1):
        if slices[0].start == 0:  # contains point on symmetry axis
            indices.append(index)
        else:
            logger = logging.getLogger(grid.__class__.__module__)
            logger.warning("Found object not located on symmetry axis")

    # determine position from binary image and scale it to real space
    pos = ndimage.center_of_mass(mask, labels, index=indices)
    pos = grid.transform(pos, "cell", "cartesian")

    # determine volume from binary image and scale it to real space
    vol_r, dz = grid.cell_volume_data
    cell_volumes = vol_r * dz
    vol = ndimage.sum(cell_volumes, labels, index=indices)

    # return an emulsion of droplets
    droplets = (
        SphericalDroplet.from_volume(np.array([0, 0, p[2]]), v)
        for p, v in zip(pos, vol)
    )
    return Emulsion(droplets, grid=grid)


def _locate_droplets_in_mask_cylindrical(
    grid: CylindricalSymGrid, mask: np.ndarray
) -> Emulsion:
    """locate droplets in a data set on a (periodic) cylindrical grid

    This function locates droplets respecting periodic boundary conditions.

    Args:
        mask (:class:`~numpy.ndarray`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    assert np.all(mask.shape == grid.shape)

    if grid.periodic[1]:
        # locate droplets respecting periodic boundary conditions in z-direction

        # pad the array to simulate periodic boundary conditions
        dim_r, dim_z = grid.shape
        z_min, z_max = grid.axes_bounds[1]
        mask_padded = np.pad(mask, [[0, 0], [dim_z, dim_z]], mode="wrap")
        assert mask_padded.shape == (dim_r, 3 * dim_z)

        # locate droplets in the extended image
        candidates = _locate_droplets_in_mask_cylindrical_single(grid, mask_padded)
        grid._logger.info(f"Found {len(candidates)} droplet candidates.")

        # keep droplets that are inside the central area
        droplets = Emulsion(grid=grid)
        for droplet in candidates:
            # correct for the additional padding of the array
            droplet.position[2] -= grid.length
            # check whether the droplet lies in the original box
            if z_min <= droplet.position[2] <= z_max:
                droplets.append(droplet)

        grid._logger.info(f"Kept {len(droplets)} central droplets.")

        # filter overlapping droplets (e.g. due to duplicates)
        droplets.remove_overlapping()

    else:
        # simply locate droplets in the mask
        droplets = _locate_droplets_in_mask_cylindrical_single(grid, mask)

    return droplets


def locate_droplets_in_mask(grid: GridBase, mask: np.ndarray) -> Emulsion:
    """locates droplets in a binary image

    This function locates droplets respecting periodic boundary conditions.

    Args:
        mask (:class:`~numpy.ndarray`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    if isinstance(grid, CartesianGrid):
        return _locate_droplets_in_mask_cartesian(grid, mask)
    elif isinstance(grid, SphericalSymGridBase):
        return _locate_droplets_in_mask_spherical(grid, mask)
    elif isinstance(grid, CylindricalSymGrid):
        return _locate_droplets_in_mask_cylindrical(grid, mask)
    elif isinstance(grid, GridBase):
        raise NotImplementedError(f"Locating droplets is not possible for grid {grid}")
    else:
        raise ValueError(f"Invalid grid {grid}")


def locate_droplets(
    phase_field: ScalarField,
    threshold: Union[float, str] = 0.5,
    *,
    minimal_radius: float = 0,
    modes: int = 0,
    interface_width: Optional[float] = None,
    refine: bool = False,
    refine_args: Optional[Dict[str, Any]] = None,
) -> Emulsion:
    """Locates droplets in the phase field

    This uses a binarized image to locate clusters of large concentration in the
    phase field, which are interpreted as droplets. Basic quantities, like
    position and size, are determined for these clusters.

    Args:
        phase_field (:class:`~pde.fields.ScalarField`):
            Scalar field that describes the concentration field of droplets
        threshold (float or str):
            The threshold for binarizing the image. If a value is given it is used
            directly. Otherwise, the following algorithms are supported:

            * `extrema`: take mean between the minimum and the maximum of the data
            * `mean`: take the mean over the entire data
            * `otsu`: use Otsu's method implemented in :func:`threshold_otsu`

            The special value `auto` currently defaults to the `extrema` method.

        minimal_radius (float):
            The smallest radius of droplets to include in the list. This can be used to
            filter out fluctuations in noisy simulations.
        modes (int):
            The number of perturbation modes that should be included. If `modes=0`,
            droplets are assumed to be spherical. Note that the mode amplitudes are only
            determined when `refine=True`.
        interface_width (float, optional):
            Interface width of the located droplets, which is also used as a starting
            value for the fitting if droplets are refined.
        refine (bool):
            Flag determining whether the droplet properties should be refined using
            fitting. This is a potentially slow procedure.
        refine_args (dict):
            Additional keyword arguments passed on to :func:`refine_droplet`. Only has
            an effect if `refine=True`.

    Returns:
        :class:`~droplets.emulsions.Emulsion`: All detected droplets
    """
    assert isinstance(phase_field, ScalarField)
    dim = phase_field.grid.dim  # dimensionality of the space

    if modes > 0 and dim not in [2, 3]:
        raise ValueError("Perturbed droplets only supported for 2d and 3d")
    if refine_args is None:
        refine_args = {}

    # determine actual threshold
    if threshold == "extrema" or threshold == "auto":
        threshold = float(phase_field.data.min() + phase_field.data.max()) / 2
    elif threshold == "mean":
        threshold = float(phase_field.data.mean())
    elif threshold == "otsu":
        threshold = threshold_otsu(phase_field.data)
    else:
        threshold = float(threshold)

    # locate droplets in thresholded image
    img_binary = phase_field.data > threshold
    candidates = locate_droplets_in_mask(phase_field.grid, img_binary)

    if minimal_radius > -np.inf:
        candidates.remove_small(minimal_radius)

    droplets = []
    for droplet in candidates:
        # check whether we need to add the interface width
        droplet_class = droplet.__class__
        args: Dict[str, NumberOrArray] = {}

        # change droplet class when interface width is given
        if interface_width is not None:
            droplet_class = DiffuseDroplet
            args["interface_width"] = interface_width

        # change droplet class when perturbed droplets are requested
        if modes > 0:
            if dim == 2:
                droplet_class = PerturbedDroplet2D
            elif dim == 3:
                droplet_class = PerturbedDroplet3D
            else:
                raise NotImplementedError(f"Dimension {dim} is not supported")
            args["amplitudes"] = np.zeros(modes)

        # recreate a droplet of the correct class
        if droplet_class != droplet.__class__:
            droplet = droplet_class.from_droplet(droplet, **args)

        # refine droplets if necessary
        if refine:
            try:
                droplet = refine_droplet(phase_field, droplet, **refine_args)
            except ValueError:
                continue  # do not add the droplet to the list
        droplets.append(droplet)

    # return droplets as an emulsion
    emulsion = Emulsion(droplets, grid=phase_field.grid)
    if minimal_radius > -np.inf:
        emulsion.remove_small(minimal_radius)
    return emulsion


def refine_droplet(
    phase_field: ScalarField,
    droplet: DiffuseDroplet,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    adjust_values: bool = False,
    least_squares_params: Optional[Dict[str, Any]] = None,
) -> DiffuseDroplet:
    """Refines droplet parameters by fitting to phase field

    This function varies droplet parameters, like position, size,
    interface width, and potential perturbation amplitudes until the overlap
    with the respective phase field region is maximized. Here, we use a
    constraint fitting routine.

    Args:
        phase_field (:class:`~pde.fields.ScalarField`):
            Phase_field that is being used to refine the droplet
        droplet (:class:`~droplets.droplets.SphericalDroplet`):
            Droplet that should be refined. This could also be a subclass of
            :class:`SphericalDroplet`
        vmin (float):
            The intensity value of the dilute phase surrounding the droplet. If `None`,
            the value will be determined automatically.
        vmax (float):
            The intensity value inside the droplet. If `None`, the value will be
            determined automatically.
        adjust_value (bool):
            Flag determining whether the intensity values will be included in the
            fitting procedure. The default value `False` implies that the intensity
            values are regarded fixed.
        least_squares_params (dict):
            Dictionary of parameters that influence the fitting; see the documentation
            of :func:`scipy.optimize.least_squares`.

    Returns:
        :class:`droplets.droplets.DiffuseDroplet`: The refined droplet as an instance of
        the argument `droplet`
    """
    assert isinstance(phase_field, ScalarField)
    if least_squares_params is None:
        least_squares_params = {}

    if not isinstance(droplet, DiffuseDroplet):
        droplet = DiffuseDroplet.from_droplet(droplet)
    if droplet.interface_width is None:
        droplet.interface_width = phase_field.grid.typical_discretization

    # enlarge the mask to also contain the shape change
    mask = droplet._get_phase_field(phase_field.grid, dtype=np.bool_)
    dilation_iterations = 1 + int(2 * droplet.interface_width)
    mask = ndimage.binary_dilation(mask, iterations=dilation_iterations)

    # apply the mask
    data_mask = phase_field.data[mask]

    # determine the coordinate constraints and only vary the free data points
    data_flat = structured_to_unstructured(droplet.data)  # unstructured data
    dtype = droplet.data.dtype
    free = np.ones(len(data_flat), dtype=np.bool_)
    free[phase_field.grid.coordinate_constraints] = False

    # determine data bounds
    l, h = droplet.data_bounds
    bounds = l[free], h[free]

    # determine the intensities outside and inside the droplet
    if vmin is None:
        vmin = np.min(data_mask)
    if vmax is None:
        vmax = np.max(data_mask)
    vrng = vmax - vmin

    if adjust_values:
        # fit intensities in addition to all droplet parameters

        # add vmin and vrng as separate fitting parameters
        parameters = np.r_[data_flat[free], vmin, vmax]
        bounds = np.r_[bounds[0], vmin - vrng, 0], np.r_[bounds[1], vmax, 3 * vrng]

        def _image_deviation(params):
            """helper function evaluating the residuals"""
            # generate the droplet
            data_flat[free] = params[:-2]
            vmin, vrng = params[-2:]
            droplet.data = unstructured_to_structured(data_flat, dtype=dtype)
            droplet.check_data()
            img = vmin + vrng * droplet._get_phase_field(phase_field.grid)[mask]
            return img - data_mask

        # do the least square optimization
        result = optimize.least_squares(
            _image_deviation, parameters, bounds=bounds, **least_squares_params
        )
        data_flat[free] = result.x[:-2]

    else:
        # fit only droplet parameters and assume all intensities fixed

        def _image_deviation(params):
            """helper function evaluating the residuals"""
            # generate the droplet
            data_flat[free] = params
            droplet.data = unstructured_to_structured(data_flat, dtype=dtype)
            droplet.check_data()
            img = vmin + vrng * droplet._get_phase_field(phase_field.grid)[mask]
            return img - data_mask

        # do the least square optimization
        result = optimize.least_squares(
            _image_deviation, data_flat[free], bounds=bounds, **least_squares_params
        )
        data_flat[free] = result.x
    droplet.data = unstructured_to_structured(data_flat, dtype=dtype)

    # normalize the droplet position
    grid = phase_field.grid
    coords = grid.transform(droplet.position, "cartesian", "grid")
    droplet.position = grid.transform(grid.normalize_point(coords), "grid", "cartesian")

    return droplet


def get_structure_factor(
    scalar_field: ScalarField,
    smoothing: Union[None, float, str] = "auto",
    wave_numbers: Union[Sequence[float], str] = "auto",
    add_zero: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the structure factor associated with a field

    Here, the structure factor is basically the power spectral density of the
    field `scalar_field` normalized so that re-gridding or rescaling the field
    does not change the result.

    Args:
        scalar_field (:class:`~pde.fields.ScalarField`):
            The scalar_field being analyzed
        smoothing (float, optional):
            Length scale that determines the smoothing of the radially averaged
            structure factor. If omitted, the full data about the discretized
            structure factor is returned. The special value `auto` calculates
            a value automatically.
        wave_numbers (list of floats, optional):
            The magnitude of the wave vectors at which the structure factor is
            evaluated. This only applies when smoothing is used. If `auto`, the
            wave numbers are determined automatically.
        add_zero (bool):
            Determines whether the value at k=0 (defined to be 1) should also be
            returned.

    Returns:
        (numpy.ndarray, numpy.ndarray): Two arrays giving the wave numbers and
        the associated structure factor
    """
    logger = logging.getLogger(__name__)

    if not isinstance(scalar_field, ScalarField):
        raise TypeError(
            "Length scales can only be calculated for scalar "
            f"fields, not {scalar_field.__class__.__name__}"
        )

    grid = scalar_field.grid
    if not isinstance(grid, CartesianGrid):
        raise NotImplementedError(
            "Structure factor can currently only be calculated for Cartesian grids"
        )
    if not all(grid.periodic):
        logger.warning(
            "Structure factor calculation assumes periodic boundary "
            "conditions, but not all grid dimensions are periodic"
        )

    # do the n-dimensional Fourier transform and calculate the structure factor
    f1 = np_fftn(scalar_field.data, norm="ortho").flat[1:]
    flat_data = scalar_field.data.flat
    sf = np.abs(f1) ** 2 / np.dot(flat_data, flat_data)

    # an alternative calculation of the structure factor is
    #    f2 = np_ifftn(scalar_field.data, norm='ortho').flat[1:]
    #    sf = (f1 * f2).real
    #    sf /= (scalar_field.data**2).sum()
    # but since this involves two FFT, it is probably slower

    # determine the (squared) components of the wave vectors
    k2s = [
        np.fft.fftfreq(grid.shape[i], d=grid.discretization[i]) ** 2
        for i in range(grid.dim)
    ]
    # calculate the magnitude
    k_mag = np.sqrt(reduce(np.add.outer, k2s)).flat[1:]

    no_wavenumbers = wave_numbers is None or (
        isinstance(wave_numbers, str) and wave_numbers == "auto"
    )

    if smoothing is not None and smoothing != "none":
        # construct the smoothed function of the structure factor
        if smoothing == "auto":
            smoothing = k_mag.max() / 128
        smoothing = float(smoothing)  # type: ignore
        sf_smooth = SmoothData1D(k_mag, sf, sigma=smoothing)

        if no_wavenumbers:
            # determine the wave numbers at which to evaluate it
            k_min = 2 / grid.cuboid.size.max()
            k_max = k_mag.max()
            k_mag = np.linspace(k_min, k_max, 128)

        else:
            k_mag = np.array(wave_numbers)

        # obtain the smoothed values at these points
        sf = sf_smooth(k_mag)

    elif not no_wavenumbers:
        logger.warning(
            "Argument `wave_numbers` is only used when `smoothing` is enabled."
        )

    if add_zero:
        sf = np.r_[1, sf]
        k_mag = np.r_[0, k_mag]

    return k_mag, sf


def get_length_scale(
    scalar_field: ScalarField, method: str = "structure_factor_maximum", **kwargs
) -> Union[float, Tuple[float, Any]]:
    """Calculates a length scale associated with a phase field

    Args:
        scalar_field (:class:`~pde.fields.ScalarField`):
            The scala field being analyzed
        method (str):
            A string determining which method is used to calculate the length scale.
            Valid options are `structure_factor_maximum` (numerically determine the
            maximum in the structure factor) and `structure_factor_mean` (calculate the
            mean of the structure factor).

    Additional supported keyword arguments depend on the chosen method. For instance,
    the methods involving the structure factor allow for a boolean flag `full_output`,
    which also returns the actual structure factor. The method
    `structure_factor_maximum` also allows for some smoothing of the radially averaged
    structure factor. If the parameter `smoothing` is set to `None` the amount of
    smoothing is determined automatically from the typical discretization of the
    underlying grid. For the method `droplet_detection`, additional arguments are
    forwarded to :func:`locate_droplets`.

    Returns:
        float: The determine length scale

        If `full_output = True`, a tuple with the length scale and an additional
        object with further information is returned.

    See Also:
        :class:`~droplets.trackers.LengthScaleTracker`: Tracker measuring length scales
    """
    logger = logging.getLogger(__name__)

    if method == "structure_factor_mean" or method == "structure_factor_average":
        # calculate the structure factor
        k_mag, sf = get_structure_factor(scalar_field)
        length_scale = np.sum(sf) / np.sum(k_mag * sf)

        if kwargs.pop("full_output", False):
            return length_scale, sf

    elif method == "structure_factor_maximum" or method == "structure_factor_peak":
        # calculate the structure factor
        k_mag, sf = get_structure_factor(scalar_field, smoothing=None, add_zero=True)

        # smooth the structure factor
        if kwargs.pop("smoothing", None) is None:
            smoothing = 0.01 * scalar_field.grid.typical_discretization
        sf_smooth = SmoothData1D(k_mag, sf, sigma=smoothing)

        # find the maximum
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # determine maximum (excluding k=0)
            max_est = k_mag[1 + np.argmax(sf[1:])]
            bracket = np.array([0.2, 1, 5]) * max_est
            logger.debug(f"Search maximum of structure factor in interval {bracket}")
            try:
                result = optimize.minimize_scalar(
                    lambda x: -sf_smooth(x), bracket=bracket
                )
            except Exception:
                logger.exception("Could not determine maximal structure factor")
                length_scale = math.nan
            else:
                if not result.success:
                    logger.warning(
                        "Maximization of structure factor resulted in the following "
                        f"message: {result.message}"
                    )
                length_scale = 1 / result.x

        if kwargs.pop("full_output", False):
            return length_scale, sf_smooth

    elif method == "droplet_detection":
        # calculate the length scale from detected droplets
        droplets = locate_droplets(scalar_field, **kwargs)
        kwargs = {}  # clear kwargs, so no warning is raised

        # get the axes along which droplets can be placed
        grid = scalar_field.grid
        axes = set(range(grid.dim)) - set(grid.coordinate_constraints)
        volume = 1.0
        for ax in axes:
            volume *= grid.axes_bounds[ax][1] - grid.axes_bounds[ax][0]

        volume_per_droplet = volume / len(droplets)
        length_scale = volume_per_droplet ** (1 / len(axes))

    else:
        raise ValueError(
            f"Method {method} is not defined. Valid values are `structure_factor_mean` "
            "and `structure_factor_maximum`"
        )

    if kwargs:
        # raise warning if keyword arguments remain
        logger.warning("Unused keyword arguments: %s", ", ".join(kwargs))

    # return only the length scale with out any additional information
    return length_scale  # type: ignore


__all__ = [
    "threshold_otsu",
    "locate_droplets",
    "refine_droplet",
    "get_structure_factor",
    "get_length_scale",
]
