"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import ScalarField, UnitGrid
from pde.tools.misc import skipUnlessModule

from droplets import DiffuseDroplet, Emulsion, SphericalDroplet, droplets, emulsions


def test_empty_emulsion():
    """test an emulsions without any droplets"""
    e = Emulsion([], grid=UnitGrid([2]))
    assert not e
    assert len(e) == 0
    assert e == e.copy()
    assert e is not e.copy()
    assert e.interface_width is None
    assert e.total_droplet_volume == 0
    dists = e.get_pairwise_distances()
    np.testing.assert_array_equal(dists, np.zeros((0, 0)))
    expect = {
        "count": 0,
        "radius_mean": np.nan,
        "radius_std": np.nan,
        "volume_mean": np.nan,
        "volume_std": np.nan,
    }
    assert e.get_size_statistics() == expect
    for b in [True, False]:
        np.testing.assert_array_equal(e.get_neighbor_distances(b), np.array([]))


def test_emulsion_single():
    """test an emulsions with a single droplet"""
    e = Emulsion([], grid=UnitGrid([2]))
    e.append(DiffuseDroplet([10], 3, 1))
    assert e
    assert len(e) == 1
    assert e == e.copy()
    assert e is not e.copy()
    assert e.interface_width == pytest.approx(1)
    assert e.total_droplet_volume == pytest.approx(6)
    dists = e.get_pairwise_distances()
    np.testing.assert_array_equal(dists, np.zeros((1, 1)))
    expect = {
        "count": 1,
        "radius_mean": 3,
        "radius_std": 0,
        "volume_mean": 6,
        "volume_std": 0,
    }
    assert e.get_size_statistics() == expect
    for b in [True, False]:
        np.testing.assert_array_equal(e.get_neighbor_distances(b), np.array([np.nan]))


def test_emulsion_two():
    """test an emulsions with two droplets"""
    grid = UnitGrid([30])
    e = Emulsion([DiffuseDroplet([10], 3, 1)], grid=grid)
    e1 = Emulsion([DiffuseDroplet([20], 5, 1)], grid=grid)
    e.extend(e1)
    assert e
    assert len(e) == 2
    assert e == e.copy()
    assert e is not e.copy()
    assert e.interface_width == pytest.approx(1)
    assert e.total_droplet_volume == pytest.approx(16)

    dists = e.get_pairwise_distances()
    np.testing.assert_array_equal(dists, np.array([[0, 10], [10, 0]]))
    expect = {
        "count": 2,
        "radius_mean": 4,
        "radius_std": 1,
        "volume_mean": 8,
        "volume_std": 2,
    }
    assert e.get_size_statistics() == expect

    np.testing.assert_array_equal(e.get_neighbor_distances(False), np.array([10, 10]))
    np.testing.assert_array_equal(e.get_neighbor_distances(True), np.array([2, 2]))


def test_emulsion_incompatible():
    """test incompatible droplets in an emulsion"""
    # different type
    d1 = SphericalDroplet([1], 2)
    d2 = DiffuseDroplet([1], 2, 1)
    e = Emulsion([d1, d2])
    assert len(e) == 2
    with pytest.raises(TypeError):
        e.data

    # same type
    d1 = SphericalDroplet([1], 2)
    d2 = SphericalDroplet([1, 2], 2)
    with pytest.raises(ValueError):
        e = Emulsion([d1, d2])


def test_emulsion_linked_data():
    """test whether emulsions link the data to droplets correctly"""
    d1 = SphericalDroplet([0, 0], 1)
    d2 = SphericalDroplet([1, 2], 3)
    e = Emulsion([d1, d2])
    data = e.get_linked_data()

    assert e[0] == d1
    assert e[1] == d2

    data[1] = 5
    assert e[0] == d1
    assert e[1] != d2
    np.testing.assert_array_equal(e[1]._data_array, 5)


@skipUnlessModule("h5py")
def test_emulsion_io(tmp_path):
    """test writing and reading emulsions"""
    path = tmp_path / "test_emulsion_io.hdf5"

    es = [
        Emulsion(),
        Emulsion([DiffuseDroplet([0, 1], 10, 0.5)] * 2),
        Emulsion([droplets.PerturbedDroplet2D([0, 1], 3, 1, [1, 2, 3])]),
    ]
    for e1 in es:
        e1.to_file(path)
        e2 = Emulsion.from_file(path)
        assert e1 == e2


def test_timecourse():
    """test some droplet track functions"""
    t1 = emulsions.EmulsionTimeCourse()
    for i in range(4):
        d = droplets.SphericalDroplet([i], i)
        t1.append([d], i)

    for track, length in [(t1, 4), (t1[:2], 2)]:
        assert len(track) == length
        assert track.times == list(range(length))
        assert [t for t, _ in track.items()] == list(range(length))

    assert t1 == t1[:]
    assert t1[3:3] == emulsions.EmulsionTimeCourse()

    t2 = emulsions.EmulsionTimeCourse(t1)
    assert t1 == t2
    assert t1 is not t2

    t1.clear()
    assert len(t1) == 0


@skipUnlessModule("h5py")
def test_timecourse_io(tmp_path):
    """test writing and reading emulsions time courses"""
    path = tmp_path / "test_timecourse_io.hdf5"

    e1 = Emulsion()
    e2 = Emulsion([DiffuseDroplet([0, 1], 10, 0.5)] * 2)
    tc1 = emulsions.EmulsionTimeCourse([e1, e2], times=[0, 10])

    tc1.to_file(path)
    tc2 = emulsions.EmulsionTimeCourse.from_file(path, progress=False)
    assert tc1.times == tc2.times
    assert tc1.emulsions == tc2.emulsions
    assert len(tc2) == 2


def test_emulsion_plotting():
    """test plotting emulsions"""
    # 1d emulsion
    e1 = Emulsion([DiffuseDroplet([1], 10, 0.5)] * 2)
    with pytest.raises(NotImplementedError):
        e1.plot()

    # 2d emulsion
    field = ScalarField(UnitGrid([10, 10], periodic=True))
    es = [
        Emulsion([DiffuseDroplet([0, 1], 10, 0.5)] * 2),
        Emulsion([droplets.PerturbedDroplet2D([0, 1], 3, 1, [1, 2, 3, 4])]),
    ]
    for e2 in es:
        e2.plot()
        e2.plot(field=field, repeat_periodically=True)
        e2.plot(color_value=lambda droplet: droplet.radius)

    # 3d emulsion
    field = ScalarField(UnitGrid([5, 5, 5], periodic=True))
    e3 = Emulsion([DiffuseDroplet([0, 1, 2], 10, 0.5)] * 2)
    e3.plot()
    e3.plot(field=field)

    e3 = Emulsion([droplets.PerturbedDroplet3D([0, 1, 2], 3, 1, [1, 2, 3, 4, 5, 6])])
    with pytest.raises(NotImplementedError):
        e3.plot()

    with pytest.raises(NotImplementedError):
        Emulsion().plot()


def test_remove_overlapping():
    """test that removing overlapping droplets works"""
    e = Emulsion([SphericalDroplet([0, 1], 2), SphericalDroplet([1, 1], 2)])
    assert len(e) == 2
    e.remove_overlapping()
    assert len(e) == 1
