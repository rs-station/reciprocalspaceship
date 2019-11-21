import unittest
from os import listdir
import re
import reciprocalspaceship as rs


class TestSymmetryOps(unittest.TestCase):
    def test_hkl_to_asu(self):
        datadir = 'data/'
        files = [datadir + i for i in listdir(datadir) if re.match(r'.*(?<!_p1).mtz$', i)] 
        for inFN in files:
            x = rs.read_mtz(inFN)
            y = rs.read_mtz(inFN[:-4] + '_p1.mtz')
            y.spacegroup = x.spacegroup
            yasu = y.hkl_to_asu()
            print(f"{inFN}")
            print(f"{x.spacegroup}")
            print(f"Reference setting = {x.spacegroup.is_reference_setting()}")
            print(f"\t|merged - mapped| = {len(x.index.difference(yasu.index))}")
            print(f"\t|mapped - merged| = {len(yasu.index.difference(x.index))}")
            #There should be the same indices in merged and p1 mapped to asu
            #self.assertEqual(len(p1.index.difference(mapped_to_asu.index)), 0)
            #self.assertEqual(len(mapped_to_asu.index.difference(p1.index)), 0)

            #Check phase equivalence


            #Now check complex structure factor equivalence
            #Fmerged = merged['FMODEL'].values.astype(float) * np.exp(1j*np.deg2rad(merged['PHIFMODEL'].values.astype(float)))
            #Fp1 = p1['FMODEL'].values.astype(float) * np.exp(1j*np.deg2rad(p1['PHIFMODEL'].values.astype(float)))
            #self.assertAlmostEqual(Fmerged, Fp1)

if __name__ == '__main__':
    unittest.main()
