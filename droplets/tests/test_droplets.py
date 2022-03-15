"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import integrate

from pde.grids import UnitGrid
from pde.tools.misc import skipUnlessModule

from droplets import droplets


def test_simple_droplet():
    """test a given simple droplet"""
    d = droplets.SphericalDroplet((1, 2), 1)
    assert d.surface_area == pytest.approx(2 * np.pi)
    np.testing.assert_allclose(d.interface_position(0), [2, 2])
    np.testing.assert_allclose(d.interface_position([0]), [[2, 2]])

    d.volume = 3
    assert d.volume == pytest.approx(3)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_random_droplet(dim):
    """tests simple droplet"""
    pos = np.random.uniform(0, 10, dim)
    radius = np.random.uniform(2, 3)
    d1 = droplets.SphericalDroplet(pos, radius)
    d2 = droplets.SphericalDroplet(np.zeros(dim), radius)
    d2.position = pos

    assert d1.dim == dim
    assert d1.volume > 0
    assert d1.surface_area > 0
    assert d1 == d2

    d3 = d1.copy()
    assert d1 == d3
    assert d1 is not d3

    vol = np.random.uniform(10, 30)
    d2.volume = vol
    assert d2.volume == pytest.approx(vol)

    f = d1.get_phase_field(UnitGrid([10] * dim), vmin=0.2, vmax=0.8, label="test")
    assert f.label == "test"
    assert np.all(f.data >= 0.2)
    assert np.all(f.data <= 0.8)
    assert np.any(f.data == 0.2)
    assert np.any(f.data == 0.8)


def test_perturbed_droplet_2d():
    """test methods of perturbed droplets in 2d"""
    d = droplets.PerturbedDroplet2D([0, 1], 1, 0.1, [0.0, 0.1, 0.2])
    d.volume
    d.interface_distance(0.1)
    d.interface_position(0.1)
    d.interface_curvature(0.1)


def test_perturbed_droplet_3d():
    """test methods of perturbed droplets in 2d"""
    d = droplets.PerturbedDroplet3D([0, 1, 2], 1, 0.1, [0.0, 0.1, 0.2, 0.3])
    d.volume_approx
    d.interface_distance(0.1, 0.2)
    d.interface_position(0.1, 0.2)
    d.interface_curvature(0.1, 0.2)


def test_perturbed_volume():
    """test volume calculation of perturbed droplets"""
    pos = np.random.randn(2)
    radius = 1 + np.random.random()
    amplitudes = np.random.uniform(-0.2, 0.2, 6)
    d = droplets.PerturbedDroplet2D(pos, radius, 0, amplitudes)

    def integrand(φ):
        r = d.interface_distance(φ)
        return 0.5 * r**2

    vol = integrate.quad(integrand, 0, 2 * np.pi)[0]
    assert vol == pytest.approx(d.volume)

    vol = np.random.uniform(1, 2)
    d.volume = vol
    assert vol == pytest.approx(d.volume)

    pos = np.random.randn(3)
    radius = 1 + np.random.random()
    d = droplets.PerturbedDroplet3D(pos, radius, 0, np.zeros(7))
    assert d.volume == pytest.approx(4 * np.pi / 3 * radius**3)


def test_surface_area():
    """test surface area calculation of droplets"""
    # perturbed 2d droplet
    R0 = 3
    amplitudes = np.random.uniform(-1e-2, 1e-2, 6)

    # unperturbed droplets
    d1 = droplets.SphericalDroplet([0, 0], R0)
    d2 = droplets.PerturbedDroplet2D([0, 0], R0)
    assert d1.surface_area == pytest.approx(d2.surface_area)
    assert d2.surface_area == pytest.approx(d2.surface_area_approx)

    # perturbed droplet
    d1 = droplets.SphericalDroplet([0, 0], R0)
    d2 = droplets.PerturbedDroplet2D([0, 0], R0, amplitudes=amplitudes)
    assert d1.surface_area != pytest.approx(d2.surface_area)
    assert d2.surface_area == pytest.approx(d2.surface_area_approx, rel=1e-4)


def test_curvature():
    """test interface curvature calculation"""
    # spherical droplet
    for dim in range(1, 4):
        d = droplets.SphericalDroplet(np.zeros(dim), radius=np.random.uniform(1, 4))
        assert d.interface_curvature == pytest.approx(1 / d.radius)

    # perturbed 2d droplet
    R0 = 3
    epsilon = 0.1
    amplitudes = epsilon * np.array([0.1, 0.2, 0.3, 0.4])

    def curvature_analytical(φ):
        """analytical expression for curvature"""
        radius = (
            3.0
            * (
                5.0 * (40.0 + 27.0 * epsilon**2.0)
                + epsilon
                * (
                    40.0 * (4.0 * np.cos(2.0 * φ) + np.sin(φ))
                    + np.cos(φ) * (80.0 + 66.0 * epsilon + 240.0 * np.sin(φ))
                    - epsilon
                    * (
                        10.0 * np.cos(3.0 * φ)
                        + 21.0 * np.cos(4.0 * φ)
                        - 12.0 * np.sin(φ)
                        + 20.0 * np.sin(3.0 * φ)
                        + 72.0 * np.sin(4.0 * φ)
                    )
                )
            )
            ** (3.0 / 2.0)
            / (
                10.0
                * np.sqrt(2.0)
                * (
                    200.0
                    + 60.0
                    * epsilon
                    * (
                        2.0 * np.cos(φ)
                        + 8.0 * np.cos(2.0 * φ)
                        + np.sin(φ)
                        + 6.0 * np.sin(2.0 * φ)
                    )
                    + epsilon**2.0
                    * (
                        345.0
                        + 165.0 * np.cos(φ)
                        - 5.0 * np.cos(3.0 * φ)
                        - 21.0 * np.cos(4.0 * φ)
                        + 30.0 * np.sin(φ)
                        - 10.0 * np.sin(3.0 * φ)
                        - 72.0 * np.sin(4.0 * φ)
                    )
                )
            )
        )
        return 1 / radius

    d = droplets.PerturbedDroplet2D([0, 0], R0, amplitudes=amplitudes)
    φs = np.linspace(0, np.pi, 64)
    np.testing.assert_allclose(
        d.interface_curvature(φs), curvature_analytical(φs), rtol=1e-1
    )


def test_from_data():
    """test the from_data constructor"""
    for d1 in [
        droplets.SphericalDroplet((1,), 2),
        droplets.SphericalDroplet((1, 2), 3),
        droplets.PerturbedDroplet2D((1, 2), 3, 0.1, [0.1, 0.2]),
        droplets.PerturbedDroplet3D((1, 2, 3), 4, 0.1, [0.1, 0.2]),
    ]:
        d2 = d1.__class__.from_data(d1.data)
        assert d1 == d2
        assert d1 is not d2


@skipUnlessModule("h5py")
def test_triangulation_2d():
    """test the 2d triangulation of droplets"""
    d1 = droplets.SphericalDroplet([1, 3], 5)
    d2 = droplets.PerturbedDroplet2D([2, 4], 5, amplitudes=[0.1, 0.2, 0.1, 0.2])
    for drop in [d1, d2]:
        tri = drop.get_triangulation(0.1)
        l = sum(
            np.linalg.norm(tri["vertices"][i] - tri["vertices"][j])
            for i, j in tri["lines"]
        )
        assert l == pytest.approx(drop.surface_area, rel=1e-3), drop


@skipUnlessModule("h5py")
def test_triangulation_3d():
    """test the 3d triangulation of droplets"""
    d1 = droplets.SphericalDroplet([1, 2, 3], 5)
    d2 = droplets.PerturbedDroplet3D([2, 3, 4], 5, amplitudes=[0.1, 0.2, 0.1, 0.2])
    for drop in [d1, d2]:
        tri = drop.get_triangulation(1)
        vertices = tri["vertices"]
        vol = 0
        for a, b, c in tri["triangles"]:
            # calculate the total volume by adding the volumes of the tetrahedra
            mat = np.c_[vertices[a], vertices[b], vertices[c], drop.position]
            mat = np.vstack([mat, np.ones(4)])
            vol += abs(np.linalg.det(mat) / 6)
        assert vol == pytest.approx(drop.volume, rel=0.1), drop
