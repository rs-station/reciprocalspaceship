from os.path import abspath, dirname, join

import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs


def test_read_crystfel_mtz(IOtest_mtz):
    """
    rs.read_crystfel should raise ValueError when given file without
    .stream suffix
    """
    with pytest.raises(ValueError):
        rs.io.read_crystfel(IOtest_mtz)


@pytest.mark.parametrize("spacegroup", [None, 19, "P 21 21 21", gemmi.SpaceGroup(19)])
def test_read_stream(spacegroup):

    datadir = join(abspath(dirname(__file__)), "../data/crystfel")

    # Read HKL without providing cell / spacegroup
    hewl = rs.io.read_crystfel(join(datadir, "crystfel.stream"), spacegroup=spacegroup)

    assert np.array_equal(hewl.index.names, ["H", "K", "L"])
    assert "I" in hewl.columns
    assert "SigI" in hewl.columns
    assert "BATCH" in hewl.columns
    assert "s1x" in hewl.columns
    assert "s1y" in hewl.columns
    assert "s1z" in hewl.columns
    assert "BATCH" in hewl.columns
    assert "XDET" in hewl.columns
    assert "YDET" in hewl.columns
    assert isinstance(hewl, rs.DataSet)
    assert isinstance(hewl["I"], rs.DataSeries)
    assert isinstance(hewl["SigI"], rs.DataSeries)
    assert isinstance(hewl["ewald_offset"], rs.DataSeries)

    if isinstance(spacegroup, gemmi.SpaceGroup):
        assert hewl.spacegroup.xhm() == spacegroup.xhm()
    elif spacegroup is None:
        assert hewl.spacegroup is None
    else:
        assert hewl.spacegroup.xhm() == gemmi.SpaceGroup(spacegroup).xhm()

    assert hewl.cell is not None
    assert np.allclose(
        hewl.cell.parameters, np.array([79.20, 79.20, 38.00, 90.00, 90.00, 90.00])
    )  # use np.allclose to prevent stupid 1.000001 vs 1 errors of float conversion

    # chech values specific to the stream

    assert hewl["ewald_offset"].min() < 0 < hewl["ewald_offset"].max()
    assert len(hewl.BATCH.unique()) == 3  # grep -c 'Begin crystal' crystfel.stream
