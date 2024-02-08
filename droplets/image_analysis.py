"""
Functions for analyzing phase field images of emulsions.

.. autosummary::
   :nosignatures:

   locate_droplets
   refine_droplets
   refine_droplet
   get_structure_factor
   get_length_scale
   threshold_otsu

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
import logging
import math
import warnings
from collections.abc import Iterable, Sequence
from functools import reduce
from itertools import product
from typing import Any, Callable, Literal

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
from pde.grids import CartesianGrid, CylindricalSymGrid
from pde.grids.base import GridBase
from pde.grids.spherical import SphericalSymGridBase
from pde.tools.math import SmoothData1D
from pde.tools.typing import NumberOrArray

from .droplets import (
    DiffuseDroplet,
    PerturbedDroplet2D,
    PerturbedDroplet3D,
    PerturbedDroplet3DAxisSym,
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


def _locate_droplets_in_mask_cartesian(mask: ScalarField) -> Emulsion:
    """locate droplets in a (potentially periodic) data set on a Cartesian grid

    This function locates droplets respecting periodic boundary conditions.

    Args:
        mask (:class:`~pde.fields.scalar.ScalarField`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`~droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    grid = mask.grid
    cell_volume = np.prod(grid.discretization)

    # locate individual clusters in the padded image
    labels, num_labels = ndimage.label(mask.data)
    grid._logger.info(f"Found {num_labels} cluster(s) in image")
    if num_labels == 0:
        example_drop = SphericalDroplet(np.zeros(grid.dim), radius=0)
        return Emulsion.empty(example_drop)
    indices: Iterable = range(1, num_labels + 1)

    # determine position from binary image and scale it to real space
    positions = ndimage.center_of_mass(mask.data, labels, index=indices)
    # correct for the additional padding of the array
    positions = np.asarray(positions) + 0.5
    # determine volume from binary image and scale it to real space
    volumes = ndimage.sum(mask.data, labels, index=indices)
    volumes = np.asanyarray(volumes) * cell_volume

    # connect clusters linked viaperiodic boundary conditions
    for ax in np.flatnonzero(grid.periodic):  # look at all periodic axes
        # compile list of all boundary points connected along the current axis
        low: list[list[int] | np.ndarray] = []
        high: list[list[int] | np.ndarray] = []
        for a in range(grid.num_axes):
            if a == ax:
                low.append([0])
                high.append([-1])
            else:
                low.append(np.arange(grid.shape[a]))
                high.append(np.arange(grid.shape[a]))

        # iterate over all boundary points
        for l, h in zip(product(*low), product(*high)):
            i_l, i_h = labels[l], labels[h]
            if i_l > 0 and i_h > 0 and i_l != i_h:
                # boundary condition on the low side connects to that of the high side
                # -> we combine the cluster into one, setting is new position as the
                # weighted averages of the center of mass
                v_l, v_h = volumes[i_l - 1], volumes[i_h - 1]
                pos_l, pos_h = positions[i_l - 1], positions[i_h - 1]
                pos_h[ax] -= grid.shape[ax]  # wrap around the upper point
                pos = (pos_l * v_l + pos_h * v_h) / (v_l + v_h)
                # update both clusters with the new data
                positions[i_h - 1] = positions[i_l - 1] = pos
                volumes[i_h - 1] = volumes[i_l - 1] = v_l + v_h
                labels[labels == i_h] = i_l

    # determine which clusters are actually present
    indices = np.array(sorted(set(np.unique(labels)) - {0}))

    # create the list of droplets
    positions = grid.normalize_point(grid.transform(positions, "cell", "grid"))
    droplets = (
        SphericalDroplet.from_volume(position, volume)
        for position, volume in zip(positions[indices - 1], volumes[indices - 1])
    )

    # filter overlapping droplets (e.g. due to duplicates)
    emulsion = Emulsion(droplets)
    num_candidates = len(emulsion)
    if num_candidates < num_labels:
        grid._logger.info(f"Only {num_candidates} candidate(s) inside bounds")

    emulsion.remove_overlapping(grid=grid)
    if len(emulsion) < num_candidates:
        grid._logger.info(f"Only {num_candidates} candidate(s) not overlapping")

    return emulsion


def _locate_droplets_in_mask_spherical(mask: ScalarField) -> Emulsion:
    """locates droplets in a binary data set on a spherical grid

    Args:
        mask (:class:`~pde.fields.scalar.ScalarField`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`~droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    grid = mask.grid
    # locate clusters in the binary image
    labels, num_labels = ndimage.label(mask.data)
    if num_labels == 0:
        example_drop = SphericalDroplet(np.zeros(grid.dim), radius=0)
        return Emulsion.empty(example_drop)

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
        return Emulsion([droplet])
    else:
        example_drop = SphericalDroplet(np.zeros(grid.dim), radius=0)
        return Emulsion.empty(example_drop)


class _SpanningDropletSignal(RuntimeError):
    """exception signaling that an untypical droplet spanning the system was found"""

    ...


def _locate_droplets_in_mask_cylindrical_single(
    grid: CylindricalSymGrid, mask: np.ndarray
) -> Emulsion:
    """locate droplets in a data set on a single cylindrical grid

    Args:
        grid:
            The cylindrical grid
        mask (:class:`~numpy.ndarray`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`~droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    # locate the individual clusters
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        example_drop = SphericalDroplet(np.zeros(grid.dim), radius=0)
        return Emulsion.empty(example_drop)

    # locate clusters on the symmetry axis
    object_slices = ndimage.find_objects(labels)
    indices = []
    for index, slices in enumerate(object_slices, 1):
        if slices[0].start == 0:  # contains point on symmetry axis
            indices.append(index)
            if slices[1].start == 0 and slices[1].stop > grid.shape[1]:
                # the "droplet" extends the entire z-axis
                raise _SpanningDropletSignal
        else:
            logger = logging.getLogger(grid.__class__.__module__)
            logger.warning("Found object not located on symmetry axis")

    # determine position from binary image and scale it to real space
    pos = ndimage.center_of_mass(mask, labels, index=indices)
    pos = grid.transform(pos, "cell", "cartesian")

    # determine volume from binary image and scale it to real space
    vol_r, dz = grid.cell_volume_data
    cell_volumes = np.outer(vol_r, dz)
    try:
        vol = ndimage.sum_labels(cell_volumes, labels, index=indices)
    except AttributeError:
        # fall-back for older versions of scipy
        vol = ndimage.sum(cell_volumes, labels, index=indices)

    # return an emulsion of droplets
    droplets = (
        SphericalDroplet.from_volume(np.array([0, 0, p[2]]), v)
        for p, v in zip(pos, vol)
    )
    return Emulsion(droplets)


def _locate_droplets_in_mask_cylindrical(mask: ScalarField) -> Emulsion:
    """locate droplets in a data set on a (periodic) cylindrical grid

    This function locates droplets respecting periodic boundary conditions.

    Args:
        mask (:class:`~pde.fields.scalar.ScalarField`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`~droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    assert isinstance(mask.grid, CylindricalSymGrid)
    grid = mask.grid

    if grid.periodic[1]:
        # locate droplets respecting periodic boundary conditions in z-direction

        # pad the array to simulate periodic boundary conditions
        dim_r, dim_z = grid.shape
        z_min, z_max = grid.axes_bounds[1]
        mask_padded = np.pad(mask.data, [[0, 0], [dim_z, dim_z]], mode="wrap")
        assert mask_padded.shape == (dim_r, 3 * dim_z)

        # locate droplets in the extended image
        try:
            candidates = _locate_droplets_in_mask_cylindrical_single(grid, mask_padded)
        except _SpanningDropletSignal:
            pass
        else:
            grid._logger.info(f"Found {len(candidates)} droplet candidates.")

            # keep droplets that are inside the central area
            droplets = Emulsion()
            for droplet in candidates:
                # correct for the additional padding of the array
                droplet.position[2] -= grid.length
                # check whether the droplet lies in the original box
                if z_min <= droplet.position[2] <= z_max:
                    droplets.append(droplet)

            grid._logger.info(f"Kept {len(droplets)} central droplets.")

            # filter overlapping droplets (e.g. due to duplicates)
            droplets.remove_overlapping()
            return droplets

    # simply locate droplets in the mask
    droplets = _locate_droplets_in_mask_cylindrical_single(mask.grid, mask.data)

    return droplets


def locate_droplets_in_mask(mask: ScalarField) -> Emulsion:
    """locates droplets in a binary image

    This function locates droplets respecting periodic boundary conditions.

    Args:
        mask (:class:`~pde.fields.scalar.ScalarField`):
            The binary image (or mask) in which the droplets are searched

    Returns:
        :class:`~droplets.emulsions.Emulsion`: The discovered spherical droplets
    """
    if isinstance(mask.grid, CartesianGrid):
        return _locate_droplets_in_mask_cartesian(mask)
    elif isinstance(mask.grid, SphericalSymGridBase):
        return _locate_droplets_in_mask_spherical(mask)
    elif isinstance(mask.grid, CylindricalSymGrid):
        return _locate_droplets_in_mask_cylindrical(mask)
    elif isinstance(mask.grid, GridBase):
        raise NotImplementedError(f"Locating droplets is not possible for {mask.grid}")
    else:
        raise ValueError(f"Invalid grid {mask.grid}")


def locate_droplets(
    phase_field: ScalarField,
    threshold: float | Literal["auto", "extrema", "mean", "otsu"] = 0.5,
    *,
    minimal_radius: float = 0,
    modes: int = 0,
    interface_width: float | None = None,
    refine: bool = False,
    refine_args: dict[str, Any] | None = None,
    num_processes: int | Literal["auto"] = 1,
) -> Emulsion:
    """Locates droplets in the phase field

    This uses a binarized image to locate clusters of large concentration in the phase
    field, which are interpreted as droplets. Basic quantities, like position and size,
    are determined for these clusters.

    Example:
        To determine the position, radius, and interfacial width of an arbitrary
        droplet, the following call can be used

        .. code-block:: python

            emulsion = droplets.locate_droplets(
                field,
                threshold="auto",
                refine=True,
                refine_args={'vmin': None, 'vmax': None},
            )

        :code:`field` is the scalar field, in which the droplets are located. The
        `refine_args` set flexibel boundaries for the intensities inside and outside
        the droplet.

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
        num_processes (int):
            Number of processes used for the refinement. If set to "auto", the number of
            processes is choosen automatically.

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
    img_binary = ScalarField(phase_field.grid, phase_field.data > threshold, dtype=bool)
    candidates = locate_droplets_in_mask(img_binary)

    if minimal_radius > -np.inf:
        candidates.remove_small(minimal_radius)

    droplets = []
    for droplet in candidates:
        # check whether we need to add the interface width
        droplet_class = droplet.__class__
        args: dict[str, NumberOrArray] = {}

        # change droplet class when interface width is given
        if interface_width is not None:
            droplet_class = DiffuseDroplet
            args["interface_width"] = interface_width

        # change droplet class when perturbed droplets are requested
        if modes > 0:
            if dim == 2:
                droplet_class = PerturbedDroplet2D
            elif dim == 3:
                if isinstance(phase_field.grid, CylindricalSymGrid):
                    droplet_class = PerturbedDroplet3DAxisSym
                else:
                    droplet_class = PerturbedDroplet3D
            else:
                raise NotImplementedError(f"Dimension {dim} is not supported")
            args["amplitudes"] = np.zeros(modes)

        # recreate a droplet of the correct class
        if droplet_class != droplet.__class__:
            droplet = droplet_class.from_droplet(droplet, **args)

        droplets.append(droplet)

    # refine droplets if necessary
    if refine:
        droplets = refine_droplets(
            phase_field, droplets, num_processes=num_processes, **refine_args
        )

    # return droplets as an emulsion
    emulsion = Emulsion(droplets)
    if minimal_radius > -np.inf:
        emulsion.remove_small(minimal_radius)
    return emulsion


def refine_droplets(
    phase_field: ScalarField,
    candidates: Iterable[DiffuseDroplet],
    *,
    num_processes: int | Literal["auto"] = 1,
    **kwargs,
) -> list[DiffuseDroplet]:
    r"""Refines many droplets by fitting to phase field

    Args:
        phase_field (:class:`~pde.fields.ScalarField`):
            Phase_field that is being used to refine the droplet
        droplets (sequence of :class:`~droplets.droplets.SphericalDroplet`):
            Droplets that should be refined.
        num_processes (int or "auto"):
            Number of processes used for the refinement. If set to "auto", the number of
            processes is choosen automatically.
        \**kwargs (dict):
            Additional keyword arguments are passed on to :func:`refine_droplet`.

    Returns:
        list of :class:`~droplets.droplets.DiffuseDroplet`:
            The refined droplets
    """

    if num_processes == 1:
        # refine droplets serially in this process
        droplets = [
            drop
            for candidate in candidates
            if (drop := refine_droplet(phase_field, candidate, **kwargs)) is not None
        ]

    else:
        # use multiprocessing to refine droplets
        from concurrent.futures import ProcessPoolExecutor

        _refine_one: Callable[[DiffuseDroplet], DiffuseDroplet] = functools.partial(
            refine_droplet, phase_field, **kwargs
        )

        max_workers = None if num_processes == "auto" else num_processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            droplets = [
                candidate_refined
                for candidate_refined in executor.map(_refine_one, candidates)
                if candidate_refined is not None
            ]

    return droplets


def refine_droplet(
    phase_field: ScalarField,
    droplet: DiffuseDroplet,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    adjust_values: bool = False,
    tolerance: float | None = None,
    least_squares_params: dict[str, Any] | None = None,
) -> DiffuseDroplet:
    """Refines droplet parameters by fitting to phase field

    This function varies droplet parameters, like position, size, interface width, and
    potential perturbation amplitudes until the overlap with the respective phase field
    region is maximized. Here, we use a constraint fitting routine.

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
        tolerance (float, optional):
            Sets the three tolerance values `ftol`, `xtol`, and `gtol` of the
            :func:`scipy.optimize.least_squares`, unless they are specified in detail by
            the `least_squares_params` argument.
        least_squares_params (dict):
            Dictionary of parameters that influence the fitting; see the documentation
            of :func:`scipy.optimize.least_squares`.

    Returns:
        :class:`~droplets.droplets.DiffuseDroplet`:
            The refined droplet as an instance of the argument `droplet`
    """
    assert isinstance(phase_field, ScalarField)
    if least_squares_params is None:
        least_squares_params = {}
    if tolerance is not None:
        for key in ["ftol", "xtol", "gtol"]:
            least_squares_params.setdefault(key, tolerance)

    if not isinstance(droplet, DiffuseDroplet):
        droplet = DiffuseDroplet.from_droplet(droplet)
    if droplet.interface_width is None:
        droplet.interface_width = phase_field.grid.typical_discretization

    # enlarge the mask to also contain the shape change
    mask = droplet._get_phase_field(phase_field.grid, dtype=bool)
    dilation_iterations = 1 + int(2 * droplet.interface_width)
    mask = ndimage.binary_dilation(mask, iterations=dilation_iterations)

    # apply the mask
    data_mask = phase_field.data[mask]

    # determine the coordinate constraints and only vary the free data points
    data_flat = structured_to_unstructured(droplet.data)  # unstructured data
    dtype = droplet.data.dtype
    free: np.ndarray = np.ones(len(data_flat), dtype=bool)
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
    smoothing: None | float | Literal["auto", "none"] = "auto",
    wave_numbers: Sequence[float] | Literal["auto"] = "auto",
    add_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Calculates the structure factor associated with a field

    Here, the structure factor is basically the power spectral density of the field
    `scalar_field` normalized so that re-gridding or rescaling the field does not change
    the result.

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
        (numpy.ndarray, numpy.ndarray): Two arrays giving the wave numbers and the
        associated structure factor. Wave numbers :math:`k` are related to distances by
        :math:`2\pi/k`.
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

    # determine the (squared) components of the wave vectors.
    # Note that `fftfreq` defines the wave number in cycles per unit of the sample
    # spacing, so we need to scale lengths by one over 2π.
    k2s = [
        np.fft.fftfreq(grid.shape[i], d=grid.discretization[i] / (2 * np.pi)) ** 2
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
    scalar_field: ScalarField,
    method: Literal[
        "structure_factor_mean", "structure_factor_maximum", "droplet_detection"
    ] = "structure_factor_maximum",
    **kwargs,
) -> float | tuple[float, Any]:
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

    See Also:
        :class:`~droplets.trackers.LengthScaleTracker`: Tracker measuring length scales
    """
    logger = logging.getLogger(__name__)

    if method == "structure_factor_mean" or method == "structure_factor_average":
        # calculate the structure factor
        k_mag, sf = get_structure_factor(scalar_field)
        length_scale = 2 * np.pi * np.sum(sf) / np.sum(k_mag * sf)

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
                length_scale = 2 * np.pi / result.x

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
