'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import tempfile
import os

import pytest
import numpy as np

from pde import (UnitGrid, CartesianGrid, ScalarField, CahnHilliardPDE,
                 DiffusionPDE)
from pde.tools.misc import skipUnlessModule
    
from .. import SphericalDroplet, LengthScaleTracker
from ..emulsions import EmulsionTimeCourse
    
    

@skipUnlessModule("h5py")
def test_emulsion_tracker():
    """ test using the emulsions tracker """
    fp = tempfile.NamedTemporaryFile(suffix='.hdf5')
            
    d = SphericalDroplet([4, 4], 3)
    c = d.get_phase_field(UnitGrid([8, 8]))

    pde = CahnHilliardPDE()

    e1 = EmulsionTimeCourse()
    tracker = e1.tracker(filename=fp.name)
    pde.solve(c, t_range=1, dt=1e-3, backend='numpy', tracker=tracker)
    e2 = EmulsionTimeCourse.from_file(fp.name, progress=False)
    
    assert e1 == e2
    assert len(e1) == 2
    assert len(e1[0]) == 1  # found a single droplet
    assert os.stat(fp.name).st_size > 0  # wrote some result
    
    

def test_length_scale_tracker():
    """ test the length scale tracker """
    grid = CartesianGrid([[0, 10 * np.pi]], 64, periodic=True)
    field = ScalarField.from_expression(grid, 'sin(2 * x)')
    
    pde = DiffusionPDE()
    fp = tempfile.NamedTemporaryFile(suffix='.json')
    tracker = LengthScaleTracker(0.05, filename=fp.name)
    pde.solve(field, t_range=0.1, backend='numpy', tracker=tracker)
    
    for ls in tracker.length_scales:
        assert ls == pytest.approx(np.pi, rel=1e-3)
    assert os.stat(fp.name).st_size > 0  # wrote some result
        