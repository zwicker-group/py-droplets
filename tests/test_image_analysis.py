"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from numpy.lib.recfunctions import structured_to_unstructured

from pde import ScalarField
from pde.grids import (
    CartesianGrid,
    CylindricalSymGrid,
    PolarSymGrid,
    SphericalSymGrid,
    UnitGrid,
)

from droplets import image_analysis
from droplets.droplets import (
    DiffuseDroplet,
    PerturbedDroplet2D,
    PerturbedDroplet3D,
    SphericalDroplet,
)
from droplets.emulsions import Emulsion


@pytest.mark.parametrize("size", [16, 17])
@pytest.mark.parametrize("periodic", [True, False])
def test_localization_sym_unit(size, periodic, rng):
    """tests simple droplets localization in 2d"""
    pos = rng.random(2) * size
    radius = rng.uniform(2, 5)
    width = rng.uniform(1, 2)
    d1 = DiffuseDroplet(pos, radius, interface_width=width)

    grid = UnitGrid((size, size), periodic=periodic)
    field = d1.get_phase_field(grid)

    emulsion = image_analysis.locate_droplets(field, refine=True)
    assert len(emulsion) == 1
    d2 = emulsion[0]

    np.testing.assert_almost_equal(d1.position, d2.position)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-4)
    assert d1.interface_width == pytest.approx(d2.interface_width)

    emulsion = image_analysis.locate_droplets(field, minimal_radius=size)
    assert len(emulsion) == 0

    emulsion = image_analysis.locate_droplets(ScalarField(grid))
    assert len(emulsion) == 0


@pytest.mark.parametrize("periodic", [True, False])
def test_localization_sym_rect(periodic, rng):
    """tests simple droplets localization in 2d with a rectangular grid"""
    size = 16

    pos = rng.uniform(-4, 4, size=2)
    radius = rng.uniform(2, 5)
    width = rng.uniform(0.5, 1.5)
    d1 = DiffuseDroplet(pos, radius, interface_width=width)

    a = rng.random(2) - size / 2
    b = rng.random(2) + size / 2
    grid = CartesianGrid(np.c_[a, b], 3 * size, periodic=periodic)
    field = d1.get_phase_field(grid)

    emulsion = image_analysis.locate_droplets(field, refine=True)
    assert len(emulsion) == 1
    d2 = emulsion[0]

    np.testing.assert_almost_equal(d1.position, d2.position)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-5)
    assert d1.interface_width == pytest.approx(d2.interface_width)

    emulsion = image_analysis.locate_droplets(ScalarField(grid))
    assert len(emulsion) == 0


@pytest.mark.parametrize("periodic", [True, False])
def test_localization_perturbed_2d(periodic, rng):
    """tests localization of perturbed 2d droplets"""
    size = 16

    pos = rng.uniform(-4, 4, size=2)
    radius = rng.uniform(2, 5)
    width = rng.uniform(0.5, 1.5)
    ampls = rng.uniform(-0.01, 0.01, size=4)
    d1 = PerturbedDroplet2D(pos, radius, interface_width=width, amplitudes=ampls)

    a = rng.random(2) - size / 2
    b = rng.random(2) + size / 2
    grid = CartesianGrid(np.c_[a, b], 2 * size, periodic=periodic)
    field = d1.get_phase_field(grid)

    emulsion = image_analysis.locate_droplets(field, refine=True, modes=d1.modes)
    assert len(emulsion) == 1
    d2 = emulsion[0]

    msg = "size=%d, periodic=%s, %s != %s" % (size, periodic, d1, d2)
    np.testing.assert_almost_equal(d1.position, d2.position, decimal=1, err_msg=msg)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-5)
    assert d1.interface_width == pytest.approx(d2.interface_width, rel=1e-3)
    np.testing.assert_allclose(d1.amplitudes[1:], d2.amplitudes[1:], rtol=0.5)


@pytest.mark.parametrize("periodic", [True, False])
def test_localization_perturbed_3d(periodic, rng):
    """tests localization of perturbed 3d droplets"""
    size = 8

    pos = rng.uniform(-2, 2, size=3)
    radius = rng.uniform(2, 3)
    width = rng.uniform(0.5, 1.5)
    ampls = rng.uniform(-0.01, 0.01, size=3)
    d1 = PerturbedDroplet3D(pos, radius, interface_width=width, amplitudes=ampls)

    a = rng.random(3) - size / 2
    b = rng.random(3) + size / 2
    grid = CartesianGrid(np.c_[a, b], 2 * size, periodic=periodic)
    assert grid.dim == 3
    field = d1.get_phase_field(grid)

    emulsion = image_analysis.locate_droplets(
        field, refine=True, modes=d1.modes, refine_args={"tolerance": 1e-6}
    )
    assert len(emulsion) == 1
    d2 = emulsion[0]

    msg = "size=%d, periodic=%s, %s != %s" % (size, periodic, d1, d2)
    np.testing.assert_almost_equal(d1.position, d2.position, decimal=1, err_msg=msg)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-4)
    assert d1.interface_width == pytest.approx(d2.interface_width, rel=1e-3)
    np.testing.assert_allclose(
        d1.amplitudes[3:], d2.amplitudes[3:], rtol=0.5, err_msg=msg
    )


def test_localization_polar(rng):
    """tests simple droplets localization in polar grid"""
    radius = rng.uniform(2, 3)
    width = rng.uniform(0.5, 1.5)
    d1 = DiffuseDroplet((0, 0), radius, interface_width=width)

    grid_radius = 6 + 2 * rng.random()
    grid = PolarSymGrid(grid_radius, 16)
    field = d1.get_phase_field(grid)

    emulsion = image_analysis.locate_droplets(field, refine=True)
    assert len(emulsion) == 1
    d2 = emulsion[0]

    np.testing.assert_almost_equal(d1.position, d2.position, decimal=5)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-5)
    assert d1.interface_width == pytest.approx(d2.interface_width, rel=1e-5)

    emulsion = image_analysis.locate_droplets(ScalarField(grid))
    assert len(emulsion) == 0


def test_localization_spherical(rng):
    """tests simple droplets localization in spherical grid"""
    radius = rng.uniform(2, 3)
    width = rng.uniform(0.5, 1.5)
    d1 = DiffuseDroplet((0, 0, 0), radius, interface_width=width)

    grid_radius = 6 + 2 * rng.random()
    grid = SphericalSymGrid(grid_radius, 16)
    field = d1.get_phase_field(grid)

    emulsion = image_analysis.locate_droplets(field, refine=True)
    assert len(emulsion) == 1
    d2 = emulsion[0]

    np.testing.assert_almost_equal(d1.position, d2.position, decimal=5)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-5)
    assert d1.interface_width == pytest.approx(d2.interface_width, rel=1e-5)

    emulsion = image_analysis.locate_droplets(ScalarField(grid))
    assert len(emulsion) == 0


@pytest.mark.parametrize("periodic", [True, False])
def test_localization_cylindrical(periodic, rng):
    """tests simple droplets localization in cylindrical grid"""
    pos = (0, 0, rng.uniform(-4, 4))
    radius = rng.uniform(2, 3)
    width = rng.uniform(0.5, 1.5)
    d1 = DiffuseDroplet(pos, radius, interface_width=width)

    grid_radius = 6 + 2 * rng.random()
    bounds_z = rng.uniform(1, 2, size=2) * np.array([-4, 4])
    grid = CylindricalSymGrid(grid_radius, bounds_z, (16, 32), periodic_z=periodic)
    field = d1.get_phase_field(grid)

    emulsion = image_analysis.locate_droplets(field, refine=True)
    assert len(emulsion) == 1
    d2 = emulsion[0]

    np.testing.assert_almost_equal(d1.position, d2.position, decimal=5)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-5)
    assert d1.interface_width == pytest.approx(d2.interface_width)

    emulsion = image_analysis.locate_droplets(ScalarField(grid))
    assert len(emulsion) == 0


def test_localization_threshold(rng):
    """tests different localization thresholds"""
    pos = rng.random(2) * 16
    radius = rng.uniform(2, 5)
    width = rng.uniform(1, 2)
    d1 = DiffuseDroplet(pos, radius, interface_width=width)

    grid = UnitGrid((16, 16), periodic=False)
    field = d1.get_phase_field(grid)

    for threshold in [0.25, 0.75, "auto", "mean", "extrema", "otsu"]:
        emulsion = image_analysis.locate_droplets(
            field, threshold=threshold, refine=True
        )
        assert len(emulsion) == 1
        d2 = emulsion[0]

        np.testing.assert_almost_equal(d1.position, d2.position)
        assert d1.radius == pytest.approx(d2.radius, rel=1e-4)
        assert d1.interface_width == pytest.approx(d2.interface_width)


@pytest.mark.parametrize(
    "adjust_values, auto_values", [(False, False), (True, False), (True, True)]
)
def test_localization_vmin_vmax(adjust_values, auto_values):
    """tests localization of droplets with non-normalized densities"""
    # create perturbed droplet
    grid = CartesianGrid(bounds=[[-2, 2], [-2, 2]], shape=32, periodic=True)
    d1 = DiffuseDroplet([0, 0], 1, 0.2)
    field = d1.get_phase_field(grid, vmin=-0.1, vmax=0.1)

    if auto_values:
        refine_args = {"vmin": None, "vmax": None, "adjust_values": adjust_values}
    else:
        refine_args = {"vmin": -0.1, "vmax": 0.1, "adjust_values": adjust_values}

    # localize this droplet
    d2 = image_analysis.locate_droplets(
        field, threshold="auto", refine=True, refine_args=refine_args
    )[0]

    assert d1.position == pytest.approx(d2.position, rel=1e-4)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-4)
    assert d1.interface_width == pytest.approx(d2.interface_width)


def test_get_structure_factor(rng):
    """test the structure factor method"""
    grid = UnitGrid([512], periodic=True)
    k0 = rng.uniform(1, 3)
    field = ScalarField.from_expression(grid, f"sin({k0} * x)")
    k, S = image_analysis.get_structure_factor(field)
    k_max = k[S.argmax()]
    assert k_max == pytest.approx(k0, rel=5e-1)


@pytest.mark.parametrize(
    "method",
    [
        "structure_factor_mean",
        "structure_factor_maximum",
        "droplet_detection",
    ],
)
def test_get_length_scale(method):
    """test determining the length scale"""
    grid = CartesianGrid([[0, 8 * np.pi]], 64, periodic=True)
    c = ScalarField.from_expression(grid, "sin(x)")
    s = image_analysis.get_length_scale(c, method=method)
    assert s == pytest.approx(2 * np.pi, rel=0.1)


def test_get_length_scale_edge():
    """test determining the length scale for edge cases"""
    grid = CartesianGrid(bounds=[[0, 1]], shape=32, periodic=True)
    for n in range(1, 4):
        c = ScalarField.from_expression(grid, f"0.2 + 0.2*sin(2*{n}*pi*x)")
        s = image_analysis.get_length_scale(c, method="structure_factor_maximum")
        assert s == pytest.approx(1 / n, rel=1e-4)


def test_emulsion_processing():
    """test identifying emulsions in phase fields"""
    e1 = Emulsion(
        [
            DiffuseDroplet(position=[5, 6], radius=9, interface_width=1),
            DiffuseDroplet(position=[20, 19], radius=8, interface_width=1),
        ]
    )

    grid = UnitGrid([32, 32], periodic=True)
    field = e1.get_phasefield(grid)

    e2 = image_analysis.locate_droplets(field, refine=True)

    np.testing.assert_allclose(
        structured_to_unstructured(e1.data),
        structured_to_unstructured(e2.data),
        rtol=0.02,
    )


def test_structure_factor_random():
    """test the structure factor function for random input"""
    g1 = CartesianGrid([[0, 10]] * 2, 64, periodic=True)
    f1 = ScalarField.random_colored(g1, -2)

    # test invariance with re-meshing
    g2 = CartesianGrid([[0, 10]] * 2, [128, 64], periodic=True)
    f2 = f1.interpolate_to_grid(g2)

    ks = np.linspace(0.2, 3)
    k1, s1 = image_analysis.get_structure_factor(f1, wave_numbers=ks)
    k2, s2 = image_analysis.get_structure_factor(f2, wave_numbers=ks)

    np.testing.assert_equal(ks, k1)
    np.testing.assert_equal(ks, k2)
    np.testing.assert_allclose(s1, s2, atol=0.05)

    # test invariance with respect to scaling
    k2, s2 = image_analysis.get_structure_factor(100 * f1, wave_numbers=ks)
    np.testing.assert_equal(ks, k2)
    np.testing.assert_allclose(s1, s2, atol=0.05)


@pytest.mark.parametrize(
    "grid",
    [
        UnitGrid([5, 5], periodic=True),
        CylindricalSymGrid(3, (0, 3), 3, periodic_z=True),
    ],
)
def test_locating_stripes(grid):
    """check whether the locate_droplets function can deal with stripe morphologies"""
    field = ScalarField(grid, 1)
    em = image_analysis.locate_droplets(field)
    assert len(em) == 1
    assert em[0].volume == pytest.approx(grid.volume)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("pos", [0.5, 1.5, 3, 5.5, 6.5])
def test_droplets_on_periodic_grids(dim, pos):
    """check whether the locate_droplets function can deal with periodic BCs"""
    grid = UnitGrid([7] * dim, periodic=True)
    field = SphericalDroplet([pos] * dim, 3).get_phase_field(grid)
    em = image_analysis.locate_droplets(field)
    assert len(em) == 1
    assert em[0].volume == pytest.approx(field.integral)
    np.testing.assert_allclose(em[0].position, np.full(dim, pos), atol=0.1)


@pytest.mark.parametrize("num_processes", [1, 2])
def test_droplet_refine_parallel(num_processes):
    """tests droplets localization in 2d with and without multiprocessing"""
    grid = UnitGrid([32, 32])
    radii = [3, 2.7, 4.3]
    pos = [[7, 8], [9, 22], [22, 10]]
    em = Emulsion([DiffuseDroplet(p, r, interface_width=1) for p, r in zip(pos, radii)])
    field = em.get_phasefield(grid)

    emulsion = image_analysis.locate_droplets(
        field, refine=True, num_processes=num_processes
    )
    assert len(emulsion) == 3

    for droplet, p, r in zip(emulsion, pos, radii):
        np.testing.assert_almost_equal(droplet.position, p, decimal=4)
        assert droplet.radius == pytest.approx(r, rel=1e-4)
        assert droplet.interface_width == pytest.approx(1, rel=1e-4)
