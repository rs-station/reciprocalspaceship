import tempfile
from os.path import exists

import gemmi
import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize(
    "realmap",
    [
        np.zeros((10, 20, 30)),
        np.random.rand(10, 20, 30),
        np.zeros((10, 20)),
        np.zeros((10, 20, 30)).tolist(),
    ],
)
def test_write_ccp4_map(realmap):
    """Test rs.io.write_ccp4_map()"""
    mapfile = tempfile.NamedTemporaryFile(suffix=".map")
    sg = gemmi.SpaceGroup(1)
    cell = gemmi.UnitCell(10.0, 20.0, 30.0, 90.0, 90.0, 90.0)

    if not isinstance(realmap, np.ndarray) or not (realmap.ndim == 3):
        with pytest.raises(ValueError):
            rs.io.write_ccp4_map(realmap, mapfile.name, cell, sg)
        mapfile.close()
        return

    rs.io.write_ccp4_map(realmap, mapfile.name, cell, sg)
    assert exists(mapfile.name)
    mapfile.close()
