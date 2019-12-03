import unittest
import numpy as np
from os import listdir
from os.path import dirname
import re
import reciprocalspaceship as rs

try:
    from tqdm import tqdm
except:
    tqdm = iter


class TestSymmetryOps(unittest.TestCase):
    def test_hkl_to_asu(self):
        datadir = dirname(__file__) + '/data/'
        files = [datadir + i for i in listdir(datadir) if re.match(r'.*(?<!_p1).mtz$', i)] 
        for inFN in tqdm(files):
            x = rs.read_mtz(inFN)
            y = rs.read_mtz(inFN[:-4] + '_p1.mtz')
            y.spacegroup = x.spacegroup
            yasu = y.hkl_to_asu() #if not x.spacegroup.is_reference_setting(): #    from IPython import embed
            #    embed()
            #print(f"{inFN}")
            #print(f"{x.spacegroup}")
            #print(f"{x.spacegroup.basisop}")
            #print(f"Reference setting = {x.spacegroup.is_reference_setting()}")
            #print(f"\t|merged - mapped| = {len(x.index.difference(yasu.index))}")
            #print(f"\t|mapped - merged| = {len(yasu.index.difference(x.index))}")
            #print(f"\t|merged| = {len(x)}")
            #print(f"\t|mapped| = {len(yasu)}")
            #There should be the same indices in merged and p1 mapped to asu
            self.assertEqual(len(x.index.difference(yasu.index)), 0)
            self.assertEqual(len(yasu.index.difference(x.index)), 0)

            Fx    = x.loc[yasu.index, 'FMODEL'].values.astype(float) 
            Fyasu = yasu['FMODEL'].values.astype(float) 
            self.assertTrue(np.isclose(Fx, Fyasu).min())

            Phx    = x.loc[yasu.index, 'PHIFMODEL'].values.astype(float) 
            Phyasu = yasu['PHIFMODEL'].values.astype(float) 
            Sx    = Fx*np.exp(1j*np.deg2rad(Phx))
            Syasu = Fyasu*np.exp(1j*np.deg2rad(Phyasu))
            self.assertTrue(np.isclose(Sx, Syasu, rtol=1e-3).min())

if __name__ == '__main__':
    unittest.main()
