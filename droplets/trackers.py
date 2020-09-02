"""
Module defining classes for tracking droplets in simulations.

.. autosummary::
   :nosignatures:

   LengthScaleTracker
   DropletTracker
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, List, Optional, Union  # @UnusedImport

import numpy as np

from pde.fields.base import FieldBase
from pde.trackers.base import InfoDict, TrackerBase
from pde.trackers.intervals import IntervalData

from .emulsions import EmulsionTimeCourse


class LengthScaleTracker(TrackerBase):
    """Tracker that stores length scales measured in simulations

    Attributes:
        times (list):
            The time points at which the length scales are stored
        length_scales (list):
            The associated length scales
    """

    def __init__(
        self,
        interval: IntervalData = 1,
        filename: Optional[str] = None,
        method: str = "structure_factor_mean",
        source: Union[None, int, Callable] = None,
        verbose: bool = False,
    ):
        r"""
        Args:
            interval:
                |Arg_tracker_interval|
            filename (str, optional):
                Determines the file to which the data is written in JSON format
            method (str):
                Method used for determining the length scale. Methods are
                explain in the function :func:`~pde.analysis.get\_length\_scale`
            source (int or callable, optional):
                Determines how a field is extracted from `fields`. If `None`,
                `fields` is passed as is, assuming it is already a scalar field.
                This works for the simple, standard case where only a single
                ScalarField is treated. Alternatively, `source` can be an
                integer, indicating which field is extracted from an instance of
                :class:`~pde.fields.FieldCollection`. Lastly, `source` can be a
                function that takes `fields` as an argument and returns the
                desired field.
            verbose (bool):
                Determines whether errors in determining the length scales are
                logged.
        """
        super().__init__(interval=interval)
        self.length_scales: List[float] = []
        self.times: List[float] = []
        self.filename = filename
        self.method = method
        self.source = source
        self.verbose = verbose

    def handle(self, field: FieldBase, t: float):
        """handle data supplied to this tracker

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
            length = np.nan

        # store data
        self.times.append(t)
        self.length_scales.append(length)  # type: ignore

    def finalize(self, info: InfoDict = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        if self.filename:
            import json

            data = {"times": self.times, "length_scales": self.length_scales}
            with open(self.filename, "w") as fp:
                json.dump(data, fp)


class DropletTracker(TrackerBase):
    """Detect droplets in a scalar field during simulations

    This tracker is useful when only the parameters of actual droplets are
    needed, since it stores considerably less information compared to the full
    scalar field.

    Attributes:
        data (:class:`~droplets.emulsions.EmulsionTimeCourse`):
            Contains the data of the tracked droplets after the simulation is
            done.
    """

    def __init__(
        self,
        interval: IntervalData = 1,
        filename: Optional[str] = None,
        emulsion_timecourse=None,
        source: Union[None, int, Callable] = None,
        minimal_radius: float = 0,
        refine: bool = False,
        perturbation_modes: int = 0,
    ):
        """
        Args:
            interval:
                |Arg_tracker_interval|
            filename (str, optional):
                Determines the file to which the final data is written.
            emulsion_timecourse (:class:`EmulsionTimeCourse`, optional):
                Can be an instance of
                :class:`~droplets.emulsions.EmulsionTimeCourse` that is
                used to store the data.
            source (int or callable, optional):
                Determines how a field is extracted from `fields`. If `None`,
                `fields` is passed as is, assuming it is already a scalar field.
                This works for the simple, standard case where only a single
                ScalarField is treated. Alternatively, `source` can be an
                integer, indicating which field is extracted from an instance of
                :class:`~pde.fields.FieldCollection`. Lastly, `source` can be a
                function that takes `fields` as an argument and returns the
                desired field.
            minimal_radius (float):
                Minimal radius of droplets that will be retained.
            refine (bool):
                Flag determining whether the droplet coordinates should be
                refined using fitting. This is a potentially slow procedure.
            perturbation_modes (int):
                An option describing how many perturbation modes should be
                considered when refining droplets.

        """
        super().__init__(interval=interval)
        if emulsion_timecourse is None:
            self.data = EmulsionTimeCourse()
        else:
            self.data = emulsion_timecourse
        self.filename = filename
        self.source = source
        self.minimal_radius = minimal_radius
        self.refine = refine
        self.perturbation_modes = perturbation_modes

    def initialize(self, field: FieldBase, info: InfoDict = None) -> float:
        """
        Args:
            field (:class:`~pde.fields.base.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        if self.data.grid is None:
            self.data.grid = field.grid
        elif not self.data.grid.compatible_with(field.grid):
            raise RuntimeError(
                "Grid of the Emulsion is incompatible with the grid of current state"
            )

        return super().initialize(field, info)

    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

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
            minimal_radius=self.minimal_radius,
            refine=self.refine,
            modes=self.perturbation_modes,
        )
        self.data.append(emulsion, t)

    def finalize(self, info: InfoDict = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        super().finalize(info)
        if self.filename:
            self.data.to_file(self.filename)
