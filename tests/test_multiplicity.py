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


class TestMultiplicityCalculation(unittest.TestCase):
    def test_epsilon(self):
        inFN = abspath(dirname(__file__)) + '/data/epsilon_factors/epsilon_factors.txt'
        reference_data = pd.read_csv(inFN, index_col=0, header=None).T
        reference_data.set_index(['h', 'k', 'l'], inplace=True)
        H = np.vstack(reference_data.index)
        for key in tqdm(reference_data):
            sg = gemmi.generators_from_hall(key)
            eps = rs.utils.compute_structurefactor_multiplicity(H, sg)
            self.assertTrue(np.all(eps == reference_data[key].to_numpy()))

if __name__ == '__main__':
    unittest.main()
