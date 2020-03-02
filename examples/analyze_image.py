#!/usr/bin/env python3

from pde.fields import ScalarField
from droplets.image_analysis import locate_droplets

field = ScalarField.from_image('resources/emulsion.png')
emulsion = locate_droplets(field)

for droplet in emulsion:
    print(droplet)
    