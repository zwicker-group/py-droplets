#!/usr/bin/env python3

import numpy as np

from droplets import DiffuseDroplet, Emulsion

# create 10 random droplets
droplets = [
    DiffuseDroplet(
        position=np.random.uniform(0, 100, 2),
        radius=np.random.uniform(5, 10),
        interface_width=1,
    )
    for _ in range(10)
]

# remove overlapping droplets in emulsion and plot it
emulsion = Emulsion(droplets)
emulsion.remove_overlapping()
emulsion.plot(color_value=lambda droplet: droplet.radius, colorbar="Droplet radius")
