"""
Functions and classes for analyzing emulsions and droplets

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

# determine the package version
try:
    # try reading version of the automatically generated module
    from ._version import __version__
except ImportError:
    # determine version automatically from CVS information
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("droplets")
    except PackageNotFoundError:
        # package is not installed, so we cannot determine any version
        __version__ = "unknown"
    del PackageNotFoundError, version  # clean name space

from .droplet_tracks import DropletTrack, DropletTrackList
from .droplets import DiffuseDroplet, SphericalDroplet
from .emulsions import Emulsion, EmulsionTimeCourse
from .image_analysis import get_length_scale, get_structure_factor, locate_droplets
from .trackers import DropletTracker, LengthScaleTracker
