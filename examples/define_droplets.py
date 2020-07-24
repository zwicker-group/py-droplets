#!/usr/bin/env python3

from droplets import DiffuseDroplet, Emulsion, SphericalDroplet

# construct two droplets
drop1 = SphericalDroplet(position=[0, 0], radius=2)
drop2 = DiffuseDroplet(position=[6, 8], radius=3, interface_width=1)

# check whether they overlap
print(drop1.overlaps(drop2))  # prints False

# construct an emulsion and query it
e = Emulsion([drop1, drop2])
e.get_size_statistics()
