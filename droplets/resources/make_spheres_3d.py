#!/usr/bin/env python3
"""script that creates the resource with discretized surfaces of unit spheres """

from __future__ import annotations

import h5py
import numpy as np
import pygmsh

# generate all meshes
with h5py.File("spheres_3d.hdf5", "w") as f:
    num_list = set()

    # iterate over different characteristic lengths
    for length in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        # initialize the geometry
        with pygmsh.occ.Geometry() as geom:
            # set the resolution of the geometry
            geom.characteristic_length_min = 0.75 * length
            geom.characteristic_length_max = length

            # add unit ball
            ball = geom.add_ball([0, 0, 0], 1.0)

            # generate mesh using gmsh
            mesh = geom.generate_mesh(dim=2, verbose=False)
            num = len(mesh.points)

            if num not in num_list:
                # save triangulated data to hdf file
                group = f.create_group(str(num))
                dset = group.create_dataset("points", data=mesh.points)
                dset.attrs["column_names"] = ["x", "y", "z"]

                # obtain angles
                φ = np.arctan2(mesh.points[:, 1], mesh.points[:, 0])
                r = np.hypot(mesh.points[:, 0], mesh.points[:, 1])
                θ = np.arctan2(r, mesh.points[:, 2])
                dset = group.create_dataset("angles", data=np.c_[φ, θ])
                dset.attrs["column_names"] = ["phi", "theta"]

                # cells
                cells = []
                for data in mesh.cells:
                    if data.type == "triangle":
                        cells.append(data.data)
                    else:
                        pass
                        # raise RuntimeError(f"Unsupported cell type `{data.type}`")
                cells = np.concatenate(cells)
                dset = group.create_dataset("cells", data=cells)
                dset.attrs["column_names"] = ["p1", "p2", "p3"]

                num_list.add(num)

    # store the number of generated spheres
    f.attrs["num_list"] = list(sorted(num_list))
