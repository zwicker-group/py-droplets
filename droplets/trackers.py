"""Module defining classes for tracking droplets in simulations.

.. autosummary::
   :nosignatures:

   LengthScaleTracker
   DropletTracker

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Literal

from pde.fields.base import FieldBase
from pde.tools.docstrings import fill_in_docstring
from pde.trackers.base import InfoDict, InterruptData, TrackerBase

from .emulsions import EmulsionTimeCourse


class LengthScaleTracker(TrackerBase):
    """Tracker that stores length scales measured in simulations.

    Attributes:
        times (list):
            The time points at which the length scales are stored
        length_scales (list):
            The associated length scales
    """

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData = 1,
        filename: str | None = None,
        *,
        method: Literal[
            "structure_factor_mean", "structure_factor_maximum", "droplet_detection"
        ] = "structure_factor_mean",
        source: None | int | Callable = None,
        verbose: bool = False,
    ):
        r"""
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPTS}
            filename (str, optional):
                Determines the file to which the data is written in JSON format
            method (str):
                Method used for determining the length scale. Details are explained in
                the function :func:`~droplets.image_analysis.get_length_scale`.
            source (int or callable, optional):
                Determines how a field is extracted from `fields`. If `None`, `fields`
                is passed as is, assuming it is already a scalar field. This works for
                the simple, standard case where only a single
                :class:`~pde.fields.scalar.ScalarField` is treated. Alternatively,
                `source` can be an integer, indicating which field is extracted from an
                instance of :class:`~pde.fields.collection.FieldCollection`. Lastly,
                `source` can be a function that takes `fields` as an argument and
                returns the desired field.
            verbose (bool):
                Determines whether errors in determining the length scales are logged.
        """
        super().__init__(interrupts=interrupts)
        self.length_scales: list[float] = []
        self.times: list[float] = []
        self.filename = filename
        self.method = method
        self.source = source
        self.verbose = verbose

    def handle(self, field: FieldBase, t: float):
        """Handle data supplied to this tracker.

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        # determine length scale
        from pde.visualization.plotting import extract_field

        from .image_analysis import get_length_scale

        scalar_field = extract_field(field, self.source, 0)

        try:
            length = get_length_scale(scalar_field, method=self.method)  # type: ignore
        except Exception:
            if self.verbose:
                self._logger.exception("Could not determine length scale")
            length = math.nan

        # store data
        self.times.append(t)
        self.length_scales.append(length)  # type: ignore

    def finalize(self, info: InfoDict | None = None) -> None:
        """Finalize the tracker, supplying additional information.

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        if self.filename:
            import json

            data = {"times": self.times, "length_scales": self.length_scales}
            with Path(self.filename).open("w") as fp:
                json.dump(data, fp)


class DropletTracker(TrackerBase):
    """Detect droplets in a scalar field during simulations.

    This tracker is useful when only the parameters of actual droplets are needed, since
    it stores considerably less information compared to the full scalar field.
    The file written when `filename` is supplied can be read in later using
    :meth:`~droplets.emulsions.EmulsionTimeCourse.from_file`.

    Attributes:
        data (:class:`~droplets.emulsions.EmulsionTimeCourse`):
            Contains the data of the tracked droplets after the simulation is done.
    """

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData = 1,
        filename: str | None = None,
        *,
        emulsion_timecourse: EmulsionTimeCourse | None = None,
        source: None | int | Callable = None,
        threshold: float | Literal["auto", "extrema", "mean", "otsu"] = 0.5,
        minimal_radius: float = 0,
        refine: bool = False,
        refine_args: dict[str, Any] | None = None,
        perturbation_modes: int = 0,
    ):
        """

        Example:
            To track droplets and determine their position, radii, and interfacial
            widths, the following tracker can be used

            .. code-block:: python

                droplet_tracker = DropletTracker(
                    1, refine=True, refine_args={"vmin": None, "vmax": None}
                )

            :code:`field` is the scalar field, in which the droplets are located. The
            `refine_args` set flexible boundaries for the intensities inside and outside
            the droplet.

        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
            filename (str, optional):
                Determines the path to the HDF5 file to which the
                :class:`~droplets.emulsions.EmulsionTimeCourse` data is written.
            emulsion_timecourse (:class:`EmulsionTimeCourse`, optional):
                Can be an instance of :class:`~droplets.emulsions.EmulsionTimeCourse`
                that is used to store the data. If omitted, an empty class is initiated.
            source (int or callable, optional):
                Determines how a field is extracted from `fields`. If `None`, `fields`
                is passed as is, assuming it is already a scalar field. This works for
                the simple, standard case where only a single ScalarField is treated.
                Alternatively, `source` can be an integer, indicating which field is
                extracted from an instance of :class:`~pde.fields.FieldCollection`.
                Lastly, `source` can be a function that takes `fields` as an argument
                and returns the desired field.
            threshold (float or str):
                The threshold for binarizing the image. If a value is given it is used
                directly. Otherwise, the following algorithms are supported:

                * `extrema`: take mean between the minimum and the maximum of the data
                * `mean`: take the mean over the entire data
                * `otsu`: use Otsu's method implemented in :func:`~droplets.image_analysis.threshold_otsu`

                The special value `auto` currently defaults to the `extrema` method.

            minimal_radius (float):
                Minimal radius of droplets that will be retained.
            refine (bool):
                Flag determining whether the droplet coordinates should be
                refined using fitting. This is a potentially slow procedure.
            refine_args (dict):
                Additional keyword arguments passed on to
                :func:`~droplets.image_analysis.refine_droplet`. Only has an effect if
                `refine=True`.
            perturbation_modes (int):
                An option describing how many perturbation modes should be considered
                when refining droplets. Only has an effect if `refine=True`.
        """
        super().__init__(interrupts=interrupts)
        if emulsion_timecourse is None:
            self.data = EmulsionTimeCourse()
        else:
            self.data = emulsion_timecourse
        self.filename = filename
        self.source = source
        self.threshold = threshold
        self.minimal_radius = minimal_radius
        self.refine = refine
        self.refine_args = refine_args
        self.perturbation_modes = perturbation_modes

    def handle(self, field: FieldBase, t: float) -> None:
        """Handle data supplied to this tracker.

        Args:
            field (:class:`~pde.fields.base.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        from pde.visualization.plotting import extract_field

        from .image_analysis import locate_droplets

        scalar_field = extract_field(field, self.source, 0)
        emulsion = locate_droplets(
            scalar_field,  # type: ignore
            threshold=self.threshold,
            refine=self.refine,
            refine_args=self.refine_args,
            modes=self.perturbation_modes,
            minimal_radius=self.minimal_radius,
        )
        self.data.append(emulsion, t)

    def finalize(self, info: InfoDict | None = None) -> None:
        """Finalize the tracker, supplying additional information.

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        if self.filename:
            self.data.to_file(self.filename)
