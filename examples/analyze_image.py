#!/usr/bin/env python3

from pathlib import Path

from pde.fields import ScalarField

from droplets.image_analysis import locate_droplets

img_path = Path(__file__).parent / "resources" / "emulsion.png"
field = ScalarField.from_image(img_path)
emulsion = locate_droplets(field)

# visualize the result
emulsion.plot(field=field, fill=False, color="w")
