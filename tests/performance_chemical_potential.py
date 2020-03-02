#!/usr/bin/env python3

import sys
import os.path
import timeit

this_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_path, '..'))

import numpy as np  # @UnusedImport

from droplets.simulation.free_energies import *  # @UnusedWildImport



template = """
from __main__ import np, {free_energy} 
phi = np.random.random((64, 64))
out = np.empty_like(phi)
f = {free_energy}()
mu = f.make_chemical_potential(backend="{backend}")
mu(phi, out)
"""



def main():
    """ main routine testing the performance """
    num = 10000
    print(f'Reports the duration of {num} calls. Smaller is better.\n')
    
    # test several free energies
    for free_energy in ("GinzburgLandau2Components", "FloryHuggins2Components"):
        print("{}:".format(free_energy))
        # test several backends
        for backend in ['numpy', 'numba']:
            time = timeit.timeit("mu(phi, out)",
                    setup=template.format(free_energy=free_energy,
                                          backend=backend),
                    number=num
                )
            print(f"{backend}: {time:g}")
        print()



if __name__ == '__main__':
    main()
