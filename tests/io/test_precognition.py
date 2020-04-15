import unittest
from os.path import dirname, abspath, join, exists
from os import remove
import filecmp
import gemmi
import reciprocalspaceship as rs

class TestPrecognition(unittest.TestCase):

    def test_read_hkl(self):

        datadir = join(abspath(dirname(__file__)), '../data/precognition')
        cell = [78.97, 78.97, 38.25, 90., 90., 90.]
        
        # Read HKL without providing cell / spacegroup
        hewl = rs.read_precognition(join(datadir, 'hewl.hkl'))
        self.assertEqual(hewl.columns.to_list(), ["F", "SigF"])
        self.assertEqual(list(hewl.index.names), ["H", "K", "L"])
        self.assertIsInstance(hewl, rs.DataSet)
        self.assertIsInstance(hewl["F"], rs.DataSeries)
        self.assertIsNone(hewl.spacegroup)
        self.assertIsNone(hewl.cell)

        # Read HKL providing spacegroup
        hewl = rs.read_precognition(join(datadir, 'hewl.hkl'), sg=96)
        self.assertEqual(hewl.columns.to_list(), ["F", "SigF"])
        self.assertEqual(list(hewl.index.names), ["H", "K", "L"])
        self.assertIsInstance(hewl, rs.DataSet)
        self.assertIsInstance(hewl["F"], rs.DataSeries)
        self.assertEqual(hewl.spacegroup.number, 96)
        self.assertIsInstance(hewl.spacegroup, gemmi.SpaceGroup)
        self.assertIsNone(hewl.cell)

        # Read HKL providing cell
        hewl = rs.read_precognition(join(datadir, 'hewl.hkl'), *cell)
        self.assertEqual(hewl.columns.to_list(), ["F", "SigF"])
        self.assertEqual(list(hewl.index.names), ["H", "K", "L"])
        self.assertIsInstance(hewl, rs.DataSet)
        self.assertIsInstance(hewl["F"], rs.DataSeries)
        self.assertIsNone(hewl.spacegroup)
        self.assertIsInstance(hewl.cell, gemmi.UnitCell)

        # Read HKL providing cell and spacegroup
        hewl = rs.read_precognition(join(datadir, 'hewl.hkl'), *cell, sg=96)
        self.assertEqual(hewl.columns.to_list(), ["F", "SigF"])
        self.assertEqual(list(hewl.index.names), ["H", "K", "L"])
        self.assertIsInstance(hewl, rs.DataSet)
        self.assertIsInstance(hewl["F"], rs.DataSeries)
        self.assertEqual(hewl.spacegroup.number, 96)
        self.assertIsInstance(hewl.spacegroup, gemmi.SpaceGroup)
        self.assertIsInstance(hewl.cell, gemmi.UnitCell)
        
        return

    def test_read_ii(self):

        datadir = join(abspath(dirname(__file__)), '../data/precognition')
        cell = [35., 45., 99., 90., 90., 90.]
        
        # Read II without providing cell / spacegroup
        dhfr = rs.read_precognition(join(datadir, 'dhfr.mccd.ii'))
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertIsNone(dhfr.spacegroup)
        self.assertIsNone(dhfr.cell)

        # Read II providing spacegroup
        dhfr = rs.read_precognition(join(datadir, 'dhfr.mccd.ii'), sg=19)
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertEqual(dhfr.spacegroup.number, 19)
        self.assertIsInstance(dhfr.spacegroup, gemmi.SpaceGroup)
        self.assertIsNone(dhfr.cell)

        # Read II providing cell
        dhfr = rs.read_precognition(join(datadir, 'dhfr.mccd.ii'), *cell)
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertIsNone(dhfr.spacegroup)
        self.assertIsInstance(dhfr.cell, gemmi.UnitCell)

        # Read II providing cell and spacegroup
        dhfr = rs.read_precognition(join(datadir, 'dhfr.mccd.ii'), *cell, sg=19)
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertEqual(dhfr.spacegroup.number, 19)
        self.assertIsInstance(dhfr.spacegroup, gemmi.SpaceGroup)
        self.assertIsInstance(dhfr.cell, gemmi.UnitCell)
        
        return

    def test_write(self):
        
        datadir = join(abspath(dirname(__file__)), '../data/precognition')
        hewl = rs.read_precognition(join(datadir, 'hewl.hkl'))

        # Writing HKL shoul produce a file
        hewl.write_precognition("temp.hkl", "F", "SigF")
        self.assertTrue(exists("temp.hkl"))
        remove("temp.hkl")

        return

    def test_roundtrip(self):

        datadir = join(abspath(dirname(__file__)), '../data/precognition')
        hewl = rs.read_precognition(join(datadir, 'hewl.hkl'))

        # Write data, read data, write data again... shouldn't change
        hewl.write_precognition("temp.hkl", "F", "SigF")
        hewl2 = rs.read_precognition("temp.hkl")
        hewl2.write_precognition("temp2.hkl", "F", "SigF")

        self.assertTrue(hewl.equals(hewl2))
        self.assertTrue(filecmp.cmp("temp.hkl", "temp2.hkl"))

        # Clean up
        remove("temp.hkl")
        remove("temp2.hkl")
