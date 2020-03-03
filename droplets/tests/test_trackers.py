'''
Created on Jul 18, 2018

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import tempfile
import os

from pde.grids import UnitGrid
from pde.tools.misc import skipUnlessModule
from pde.pdes import CahnHilliardPDE
    
from .. import SphericalDroplet
from ..emulsions import EmulsionTimeCourse
    
    

@skipUnlessModule("h5py")
def test_emulsion_tracker():
    """ test using the emulsions tracker """
    
    fp = tempfile.NamedTemporaryFile(suffix='.hdf5')
            
    d = SphericalDroplet([4, 4], 3, interface_width=1)
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
    
    