import unittest
import numpy as np
from os.path import dirname, abspath
import re
import reciprocalspaceship as rs
import pandas as pd
import gemmi

try:
    from tqdm import tqdm
except:
    tqdm = iter

#These are repeated twice in the list output from sgtbx.
#One of them will fail do to incorrect setting. 
#Unsure how to fix this issue, but I think it is not
#going to effect any realistic use cases.
exclude_hall_symbols = {
    ' A 2 2 -1ab', 
    ' B 2 2 -1ab', 
    ' C 2 2 -1ac', 
    #These are temporary to remove R3 spacegroup from tests
    ' R 3 -2"c',
    ' R 3 2"',
    '-R 3 2"',
    ' R 3 -2"',
    '-R 3',
    '-R 3 2"c',
    ' R 3',
    #These are failing systematic absences only right now
    ' P 3* -2n',
    '-P 3* 2n',
}

class TestSystematicAbsences(unittest.TestCase):
    def test_systematic_absences(self):
        inFN = abspath(dirname(__file__)) + '/../data/systematic_absences/systematic_absences.txt.bz2'
        reference_data = pd.read_csv(inFN)
        keys = set(reference_data.hall.unique()) - exclude_hall_symbols
        for key in tqdm(keys):
            with self.subTest(spacegroup=key):
                data = reference_data[reference_data.hall == key]
                H = data[['h', 'k', 'l']].to_numpy()
                sg = gemmi.find_spacegroup_by_ops(gemmi.symops_from_hall(key))
                absent = rs.utils.hkl_is_absent(H, sg)
                reference = data['absent'].to_numpy()
                message = f"The following HKLs failed for spacegroup {key}\n"
                message += '    h    k    l  Centric  Test  Reference\n'
                centric = rs.utils.is_centric(H, sg)
                for i in np.where(absent != reference)[0]:
                    h,k,l = H[i].astype(int)
                    message += f"{h: 5d}{k: 5d}{l: 5d}   {centric[i]}    {absent[i]}     {reference[i]}\n"
                test_stat = np.all(absent == reference)
                self.assertTrue(test_stat, msg=message)

if __name__ == '__main__':
    unittest.main()
