import pytest
import unittest
from os.path import dirname, abspath, join, exists
from os import remove
import filecmp
import gemmi
import numpy as np
from pandas.testing import assert_frame_equal
import reciprocalspaceship as rs
from reciprocalspaceship.utils import in_asu

class TestMTZ(unittest.TestCase):

    def test_read(self):

        datadir = join(abspath(dirname(__file__)), '../data/fmodel')
        data = rs.read_mtz(join(datadir, '9LYZ.mtz'))
        
        # Confirm columns, indices, and metadata
        self.assertEqual(data.spacegroup.number, 96)
        self.assertEqual(data.columns.to_list(), ["FMODEL", "PHIFMODEL"])
        self.assertEqual(list(data.index.names), ["H", "K", "L"])
        self.assertIsInstance(data.spacegroup, gemmi.SpaceGroup)
        self.assertIsInstance(data.cell, gemmi.UnitCell)
        self.assertIsInstance(data, rs.DataSet)
        self.assertIsInstance(data["FMODEL"], rs.DataSeries)
        
        return
    
    def test_write(self):

        datadir = join(abspath(dirname(__file__)), '../data/fmodel')
        data = rs.read_mtz(join(datadir, '9LYZ.mtz'))

        # Missing cell should raise AttributeError
        data_missingcell = data.copy()
        data_missingcell.cell = None
        with self.assertRaises(AttributeError):
            data_missingcell.write_mtz("temp.mtz")
            
        # Missing spacegroup should raise AttributeError
        data_missingsg = data.copy()
        data_missingsg.spacegroup = None
        with self.assertRaises(AttributeError):
            data_missingsg.write_mtz("temp.mtz")

        # Writing MTZ should produce a file
        data.write_mtz("temp.mtz")
        self.assertTrue(exists("temp.mtz"))
        remove("temp.mtz")

        # Having a non-MTZType should raise AttributeError, unless flag
        # is set
        data["nonMTZ"] = 1
        with self.assertRaises(ValueError):
            data.write_mtz("temp.mtz")
        data.write_mtz("temp.mtz", skip_problem_mtztypes=True)
        self.assertTrue(exists("temp.mtz"))
        remove("temp.mtz")

        return
    
    def test_roundtrip(self):
        
        datadir = join(abspath(dirname(__file__)), '../data/fmodel')
        data = rs.read_mtz(join(datadir, '9LYZ.mtz'))

        # Write data, read data, write data again... shouldn't change
        data.write_mtz("temp.mtz")
        data2 = rs.read_mtz("temp.mtz")
        data2.write_mtz("temp2.mtz")

        self.assertTrue(data.equals(data2))
        self.assertEqual(data.spacegroup.number, data2.spacegroup.number)
        self.assertEqual(data.cell.a, data2.cell.a)
        self.assertTrue(filecmp.cmp("temp.mtz", "temp2.mtz"))

        # Clean up
        remove("temp.mtz")
        remove("temp2.mtz")
        
        return


def test_read_unmerged(data_unmerged):
    """Test rs.read_mtz() with unmerged data"""
    # Unmerged data will not be in asu, and should have a PARTIAL column
    assert not in_asu(data_unmerged.get_hkls(), data_unmerged.spacegroup).all()
    assert "PARTIAL" in data_unmerged.columns
    assert data_unmerged["PARTIAL"].dtype.name == "bool"
    assert not  "M/ISYM" in data_unmerged.columns
    assert not data_unmerged.merged


@pytest.mark.parametrize("label_centrics", [True, False])
def test_roundtrip(data_unmerged, label_centrics):
    """
    Test roundtrip of rs.read_mtz() and DataSet.write_mtz() with unmerged data
    """
    if label_centrics:
        data_unmerged.label_centrics(inplace=True)
    data_unmerged.write_mtz("temp.mtz")
    data2 = rs.read_mtz("temp.mtz")
    data2.write_mtz("temp2.mtz")

    assert filecmp.cmp("temp.mtz", "temp2.mtz")
    assert_frame_equal(data_unmerged, data2)
    assert data_unmerged.merged == data2.merged

    # Clean up
    remove("temp.mtz")
    remove("temp2.mtz")
