"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from pde.grids import UnitGrid
from pde.tools.misc import skipUnlessModule

from droplets.droplet_tracks import DropletTrack, DropletTrackList
from droplets.droplets import DiffuseDroplet, SphericalDroplet
from droplets.emulsions import Emulsion, EmulsionTimeCourse


def test_droplettrack():
    """test some droplet track functions"""
    t1 = DropletTrack()
    for i in range(4):
        t1.append(SphericalDroplet([i], i), i)

    for track, length in [(t1, 4), (t1[:2], 2)]:
        assert len(track) == length
        assert track.dim == 1
        assert track.start == 0
        assert track.end == length - 1
        assert track.times == list(range(length))
        assert [t for t, _ in track.items()] == list(range(length))
        assert isinstance(str(track), str)
        assert isinstance(repr(track), str)
        np.testing.assert_allclose(track.get_position(0), [0])
        np.testing.assert_array_equal(track.data["time"], np.arange(length))

    assert t1 == t1[:]
    assert t1[3:3] == DropletTrack()
    assert t1.time_overlaps(t1[:2])
    np.testing.assert_allclose(t1.get_radii(), np.arange(4))
    np.testing.assert_allclose(t1.get_volumes(), 2 * np.arange(4))

    t2 = DropletTrack(t1)
    assert t1 == t2
    assert t1 is not t2


@skipUnlessModule("h5py")
def test_droplettrack_io(tmp_path):
    """test writing and reading droplet tracks"""
    path = tmp_path / "test_droplettrack_io.hdf5"

    t1 = DropletTrack()
    ds = [DiffuseDroplet([0, 1], 10, 0.5)] * 2
    t2 = DropletTrack(droplets=ds, times=[0, 10])

    for t_out in [t1, t2]:
        t_out.to_file(path)
        t_in = DropletTrack.from_file(path)
        assert t_in == t_out
        assert t_in.times == t_out.times
        assert t_in.droplets == t_out.droplets


def test_droplettrack_plotting():
    """test writing and reading droplet tracks"""
    ds = [DiffuseDroplet([0, 1], 10, 0.5)] * 2
    t = DropletTrack(droplets=ds, times=[0, 10])
    t.plot("radius")
    t.plot("volume")
    t.plot_positions()
    t.plot_positions(grid=UnitGrid([5, 5], periodic=True))


def test_droplettracklist():
    """test droplet tracks"""
    t1 = DropletTrack()
    ds = [DiffuseDroplet([0, 1], 10, 0.5)] * 2
    t2 = DropletTrack(droplets=ds, times=[0, 10])
    tl = DropletTrackList([t1, t2])

    assert len(tl) == 2
    assert tl[0] == t1
    assert tl[1] == t2

    tl.remove_short_tracks()
    assert len(tl) == 1


@skipUnlessModule("h5py")
def test_droplettracklist_io(tmp_path):
    """test writing and reading droplet tracks"""
    path = tmp_path / "test_droplettracklist_io.hdf5"

    t1 = DropletTrack()
    ds = [DiffuseDroplet([0, 1], 10, 0.5)] * 2
    t2 = DropletTrack(droplets=ds, times=[0, 10])
    tl_out = DropletTrackList([t1, t2])

    tl_out.to_file(path)
    tl_in = DropletTrackList.from_file(path)
    assert tl_in == tl_out


def test_droplettracklist_plotting():
    """test plotting droplet tracks"""
    t1 = DropletTrack()
    ds = [DiffuseDroplet([0, 1], 10, 0.5)] * 2
    t2 = DropletTrack(droplets=ds, times=[0, 10])
    DropletTrackList([t1, t2]).plot()
    DropletTrackList([t1, t2]).plot_positions()


def test_conversion_from_emulsion_timecourse():
    """test converting between DropletTrackList and EmulsionTimecourse"""
    d1 = SphericalDroplet([0, 1], 5)
    d2 = SphericalDroplet([10, 15], 4)
    times = [0, 10]

    dt1 = DropletTrack([d1] * 2, times=times)
    dt2 = DropletTrack([d2] * 2, times=times)
    dtl1 = DropletTrackList([dt1, dt2])

    e = Emulsion([d1, d2])
    etc = EmulsionTimeCourse([e, e], times=times)

    dtl2 = DropletTrackList.from_emulsion_time_course(etc)
    assert dtl1 == dtl2

    dtl3 = DropletTrackList.from_emulsion_time_course(etc, method="distance")
    assert dtl1 == dtl3

    dtl4 = DropletTrackList.from_emulsion_time_course(
        etc, method="distance", max_dist=-1
    )
    assert dtl1 != dtl4
