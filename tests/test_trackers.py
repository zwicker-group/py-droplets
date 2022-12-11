"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CahnHilliardPDE, CartesianGrid, DiffusionPDE, ScalarField, UnitGrid
from pde.tools.misc import skipUnlessModule

from droplets import LengthScaleTracker, SphericalDroplet
from droplets.emulsions import EmulsionTimeCourse


@skipUnlessModule("h5py")
def test_emulsion_tracker(tmp_path):
    """test using the emulsions tracker"""
    path = tmp_path / "test_emulsion_tracker.hdf5"

    d = SphericalDroplet([4, 4], 3)
    c = d.get_phase_field(UnitGrid([8, 8]))

    pde = CahnHilliardPDE()

    e1 = EmulsionTimeCourse()
    tracker = e1.tracker(filename=path)
    pde.solve(c, t_range=1, dt=1e-3, backend="numpy", tracker=tracker)
    e2 = EmulsionTimeCourse.from_file(path, progress=False)

    assert e1 == e2
    assert len(e1) == 2
    assert len(e1[0]) == 1  # found a single droplet
    assert path.stat().st_size > 0  # wrote some result


def test_length_scale_tracker(tmp_path):
    """test the length scale tracker"""
    path = tmp_path / "test_length_scale_tracker.json"

    grid = CartesianGrid([[0, 10 * np.pi]], 64, periodic=True)
    field = ScalarField.from_expression(grid, "sin(2 * x)")

    pde = DiffusionPDE()
    tracker = LengthScaleTracker(0.05, filename=path)
    pde.solve(field, t_range=0.1, backend="numpy", tracker=tracker)

    for ls in tracker.length_scales:
        assert ls == pytest.approx(np.pi, rel=1e-3)
    assert path.stat().st_size > 0  # wrote some result
