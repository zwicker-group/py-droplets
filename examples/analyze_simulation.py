#!/usr/bin/env python3

from droplets.emulsions import EmulsionTimeCourse
from pde import CahnHilliardPDE, ScalarField, UnitGrid

field = ScalarField.random_uniform(UnitGrid([32, 32]), -1, 1)
pde = CahnHilliardPDE()

etc = EmulsionTimeCourse()
pde.solve(field, t_range=10, backend="numpy", tracker=etc.tracker())

print(etc[-1].get_size_statistics())
