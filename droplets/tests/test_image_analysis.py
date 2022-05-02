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
from droplets.droplets import DiffuseDroplet, PerturbedDroplet2D, PerturbedDroplet3D
from droplets.emulsions import Emulsion


@pytest.mark.parametrize("size", [16, 17])
@pytest.mark.parametrize("periodic", [True, False])
def test_localization_sym_unit(size, periodic):
    """tests simple droplets localization in 2d"""
    pos = np.random.random(2) * size
    radius = np.random.uniform(2, 5)
    width = np.random.uniform(1, 2)
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
def test_localization_sym_rect(periodic):
    """tests simple droplets localization in 2d with a rectangular grid"""
    size = 16

    pos = np.random.uniform(-4, 4, size=2)
    radius = np.random.uniform(2, 5)
    width = np.random.uniform(0.5, 1.5)
    d1 = DiffuseDroplet(pos, radius, interface_width=width)

    a = np.random.random(2) - size / 2
    b = np.random.random(2) + size / 2
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
def test_localization_perturbed_2d(periodic):
    """tests localization of perturbed 2d droplets"""
    size = 16

    pos = np.random.uniform(-4, 4, size=2)
    radius = np.random.uniform(2, 5)
    width = np.random.uniform(0.5, 1.5)
    ampls = np.random.uniform(-0.01, 0.01, size=4)
    d1 = PerturbedDroplet2D(pos, radius, interface_width=width, amplitudes=ampls)

    a = np.random.random(2) - size / 2
    b = np.random.random(2) + size / 2
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
def test_localization_perturbed_3d(periodic):
    """tests localization of perturbed 3d droplets"""
    size = 8

    pos = np.random.uniform(-2, 2, size=3)
    radius = np.random.uniform(2, 3)
    width = np.random.uniform(0.5, 1.5)
    ampls = np.random.uniform(-0.01, 0.01, size=3)
    d1 = PerturbedDroplet3D(pos, radius, interface_width=width, amplitudes=ampls)

    a = np.random.random(3) - size / 2
    b = np.random.random(3) + size / 2
    grid = CartesianGrid(np.c_[a, b], 2 * size, periodic=periodic)
    assert grid.dim == 3
    field = d1.get_phase_field(grid)

    emulsion = image_analysis.locate_droplets(field, refine=True, modes=d1.modes)
    assert len(emulsion) == 1
    d2 = emulsion[0]

    msg = "size=%d, periodic=%s, %s != %s" % (size, periodic, d1, d2)
    np.testing.assert_almost_equal(d1.position, d2.position, decimal=1, err_msg=msg)
    assert d1.radius == pytest.approx(d2.radius, rel=1e-4)
    assert d1.interface_width == pytest.approx(d2.interface_width, rel=1e-3)
    np.testing.assert_allclose(
        d1.amplitudes[3:], d2.amplitudes[3:], rtol=0.5, err_msg=msg
    )


def test_localization_polar():
    """tests simple droplets localization in polar grid"""
    radius = np.random.uniform(2, 3)
    width = np.random.uniform(0.5, 1.5)
    d1 = DiffuseDroplet((0, 0), radius, interface_width=width)

    grid_radius = 6 + 2 * np.random.random()
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


def test_localization_spherical():
    """tests simple droplets localization in spherical grid"""
    radius = np.random.uniform(2, 3)
    width = np.random.uniform(0.5, 1.5)
    d1 = DiffuseDroplet((0, 0, 0), radius, interface_width=width)

    grid_radius = 6 + 2 * np.random.random()
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
def test_localization_cylindrical(periodic):
    """tests simple droplets localization in cylindrical grid"""
    pos = (0, 0, np.random.uniform(-4, 4))
    radius = np.random.uniform(2, 3)
    width = np.random.uniform(0.5, 1.5)
    d1 = DiffuseDroplet(pos, radius, interface_width=width)

    grid_radius = 6 + 2 * np.random.random()
    bounds_z = np.random.uniform(1, 2, size=2) * np.array([-4, 4])
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


def test_localization_threshold():
    """tests different localization thresholds"""
    pos = np.random.random(2) * 16
    radius = np.random.uniform(2, 5)
    width = np.random.uniform(1, 2)
    d1 = DiffuseDroplet(pos, radius, interface_width=width)

    grid = UnitGrid((16, 16), periodic=False)
    field = d1.get_phase_field(grid)

    for threshold in [0.25, 0.75, "auto"]:
        emulsion = image_analysis.locate_droplets(
            field, threshold=threshold, refine=True
        )
        assert len(emulsion) == 1
        d2 = emulsion[0]

        np.testing.assert_almost_equal(d1.position, d2.position)
        assert d1.radius == pytest.approx(d2.radius, rel=1e-4)
        assert d1.interface_width == pytest.approx(d2.interface_width)


def test_get_length_scale():
    """test determining the length scale"""
    grid = CartesianGrid([[0, 8 * np.pi]], 64, periodic=True)
    c = ScalarField(grid, np.sin(grid.axes_coords[0]))
    for method in [
        "structure_factor_mean",
        "structure_factor_maximum",
        "droplet_detection",
    ]:
        s = image_analysis.get_length_scale(c, method=method)
        assert s == pytest.approx(2 * np.pi, rel=0.1)


def test_emulsion_processing():
    """test identifying emulsions in phase fields"""
    grid = UnitGrid([32, 32], periodic=True)

    e1 = Emulsion(
        [
            DiffuseDroplet(position=[5, 6], radius=9, interface_width=1),
            DiffuseDroplet(position=[20, 19], radius=8, interface_width=1),
        ],
        grid=grid,
    )
    field = e1.get_phasefield()

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
