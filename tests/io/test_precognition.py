import unittest
from os.path import dirname, abspath, join
from os import remove
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
        dhfr = rs.read_precognition(join(datadir, 'e074a_off1_001.mccd.ii'))
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertIsNone(dhfr.spacegroup)
        self.assertIsNone(dhfr.cell)

        # Read II providing spacegroup
        dhfr = rs.read_precognition(join(datadir, 'e074a_off1_001.mccd.ii'), sg=19)
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertEqual(dhfr.spacegroup.number, 19)
        self.assertIsInstance(dhfr.spacegroup, gemmi.SpaceGroup)
        self.assertIsNone(dhfr.cell)

        # Read II providing cell
        dhfr = rs.read_precognition(join(datadir, 'e074a_off1_001.mccd.ii'), *cell)
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertIsNone(dhfr.spacegroup)
        self.assertIsInstance(dhfr.cell, gemmi.UnitCell)

        # Read II providing cell and spacegroup
        dhfr = rs.read_precognition(join(datadir, 'e074a_off1_001.mccd.ii'), *cell, sg=19)
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertEqual(dhfr.spacegroup.number, 19)
        self.assertIsInstance(dhfr.spacegroup, gemmi.SpaceGroup)
        self.assertIsInstance(dhfr.cell, gemmi.UnitCell)

        # Read II providing logfile
        dhfr = rs.read_precognition(join(datadir, 'e074a_off1_001.mccd.ii'),
                                    logfile=join(datadir, 'integration.log'))
        self.assertEqual(len(dhfr.columns), 7)
        self.assertEqual(list(dhfr.index.names), ["H", "K", "L"])
        self.assertIsInstance(dhfr, rs.DataSet)
        self.assertIsInstance(dhfr["I"], rs.DataSeries)
        self.assertEqual(dhfr.spacegroup.number, 19)
        self.assertIsInstance(dhfr.spacegroup, gemmi.SpaceGroup)
        self.assertIsInstance(dhfr.cell, gemmi.UnitCell)
        self.assertEqual(dhfr.cell.a, 34.4660)
        self.assertEqual(dhfr.cell.b, 45.6000)
        self.assertEqual(dhfr.cell.c, 99.5850)
        self.assertEqual(dhfr.cell.alpha, 90.0)
        self.assertEqual(dhfr.cell.beta, 90.0)
        self.assertEqual(dhfr.cell.gamma, 90.0)
        
        return
