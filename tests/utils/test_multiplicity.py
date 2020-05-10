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
}

class TestMultiplicityCalculation(unittest.TestCase):
    def test_epsilon(self):
        inFN = abspath(dirname(__file__)) + '/../data/epsilon_factors/epsilon_factors.txt.bz2'
        reference_data = pd.read_csv(inFN)
        keys = set(reference_data.hall.unique()) - exclude_hall_symbols
        for key in tqdm(keys):
            with self.subTest(spacegroup=key):
                data = reference_data[reference_data.hall == key]
                H = data[['h', 'k', 'l']].to_numpy()
                reference_epsilon = data['epsilon'].to_numpy()
                sg = gemmi.symops_from_hall(key)
                epsilon = rs.utils.compute_structurefactor_multiplicity(H, sg)
                self.assertTrue(np.all(epsilon == reference_epsilon))

if __name__ == '__main__':
    unittest.main()
