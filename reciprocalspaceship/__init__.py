# Version number for reciprocalspaceship
def getVersionNumber():
    version = None
    try:
        from setuptools.version import metadata

        version = metadata.version("reciprocalspaceship")
    except ImportError:
        from setuptools.version import pkg_resources

        version = pkg_resources.require("reciprocalspaceship")[0].version

    return version


__version__ = getVersionNumber()

# Import submodules
from reciprocalspaceship import algorithms, stats

# Top-Level API
from reciprocalspaceship.concat import concat
from reciprocalspaceship.dataseries import DataSeries
from reciprocalspaceship.dataset import DataSet

# Add support for MTZ data types:
# see http://www.ccp4.ac.uk/html/f2mtz.html
from reciprocalspaceship.dtypes import (  # D; B; K; G; A; H; J; Y; I; R; E; P; Q; M; L; F; W
    AnomalousDifferenceDtype,
    BatchDtype,
    FriedelIntensityDtype,
    FriedelStructureFactorAmplitudeDtype,
    HendricksonLattmanDtype,
    HKLIndexDtype,
    IntensityDtype,
    M_IsymDtype,
    MTZIntDtype,
    MTZRealDtype,
    NormalizedStructureFactorAmplitudeDtype,
    PhaseDtype,
    StandardDeviationDtype,
    StandardDeviationFriedelIDtype,
    StandardDeviationFriedelSFDtype,
    StructureFactorAmplitudeDtype,
    WeightDtype,
    summarize_mtz_dtypes,
)
from reciprocalspaceship.io import (
    read_cif,
    read_crystfel,
    read_csv,
    read_mtz,
    read_pickle,
    read_precognition,
)
