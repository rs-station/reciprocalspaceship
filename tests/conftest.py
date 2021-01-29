import pytest
from os import listdir
from os.path import dirname, abspath, join
import numpy as np
import pandas as pd
import re
import reciprocalspaceship as rs
import gemmi

@pytest.fixture
def dataset_hkl():
    """
    Build DataSet for testing containing only Miller indices
    """
    hmin, hmax = -5, 5
    H = np.mgrid[hmin:hmax+1:2,hmin:hmax+1:2,hmin:hmax+1:2].reshape((3, -1)).T
    dataset = rs.DataSet({"H": H[:, 0], "K": H[:, 1], "L": H[:, 2]})
    dataset.set_index(["H", "K", "L"], inplace=True)
    return dataset

def load_dataset(datapath, as_gemmi=False):
    """
    Load dataset at given datapath. Datapath is expected to be a list of
    directories to follow.
    """
    inFN = abspath(join(dirname(__file__), *datapath))
    if as_gemmi:
        return gemmi.read_mtz_file(inFN)
    else:
        return rs.read_mtz(inFN)
    
@pytest.fixture
def data_merged():
    """
    Load HEWL diffraction data from APS 24-ID-C
    """
    datapath = ["data", "data_merged.mtz"]
    return load_dataset(datapath)

@pytest.fixture
def data_unmerged():
    """
    Load HEWL diffraction data from APS 24-ID-C
    """
    datapath = ["data", "data_unmerged.mtz"]
    return load_dataset(datapath)

@pytest.fixture(params=[
    ["data", "data_merged.mtz"],
    ["data", "data_unmerged.mtz"],
])
def data_hewl(request):
    """Yields DataSet objects for merged and unmerged MTZ files"""
    return load_dataset(request.param)

@pytest.fixture
def data_gemmi():
    """
    Load HEWL diffraction data from APS 24-ID-C as gemmi.Mtz
    """
    datapath = ["data", "fmodel", "9LYZ.mtz"]
    return load_dataset(datapath, as_gemmi=True)

