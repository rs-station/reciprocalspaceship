from os.path import abspath, dirname, join

import pytest


@pytest.fixture
def IOtest_hkl():
    """
    Path to Precognition .hkl file for I/O testing
    """
    datapath = join(abspath(dirname(__file__)), "../data/precognition")
    return join(datapath, "hewl.hkl")


@pytest.fixture
def IOtest_ii():
    """
    Path to Precognition .ii file for I/O testing
    """
    datapath = join(abspath(dirname(__file__)), "../data/precognition")
    return join(datapath, "e074a_off1_001.mccd.ii")


@pytest.fixture
def IOtest_log():
    """
    Path to Precognition integration logfile for I/O testing
    """
    datapath = join(abspath(dirname(__file__)), "../data/precognition")
    return join(datapath, "integration.log")


@pytest.fixture
def IOtest_mtz():
    """
    Path to MTZ file for I/O testing
    """
    datapath = join(abspath(dirname(__file__)), "../data/fmodel")
    return join(datapath, "9LYZ.mtz")


@pytest.fixture
def IOtest_cif():
    """
    Path to CIF reflection file for I/O testing
    """
    datapath = join(abspath(dirname(__file__)), "../data/fmodel")
    return join(datapath, "1CTJ-sf.cif")
