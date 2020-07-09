# Version number for reciprocalspaceship
def getVersionNumber():
    import pkg_resources
    version = pkg_resources.require("reciprocalspaceship")[0].version
    return version
__version__ = getVersionNumber()

# Top-Level API
from .dataset import DataSet
from .dataseries import DataSeries
from .io import read_mtz, read_precognition
from .dtypes import summarize_mtz_dtypes

# Add support for MTZ data types:
# see http://www.ccp4.ac.uk/html/f2mtz.html
from .dtypes import (
    HKLIndexDtype,                           # H
    IntensityDtype,                          # J
    StructureFactorAmplitudeDtype,           # F
    AnomalousDifferenceDtype,                # D
    StandardDeviationDtype,                  # Q
    FriedelStructureFactorAmplitudeDtype,    # G
    StandardDeviationFriedelSFDtype,         # L
    FriedelIntensityDtype,                   # K
    StandardDeviationFriedelIDtype,          # M
    NormalizedStructureFactorAmplitudeDtype, # E
    PhaseDtype,                              # P
    WeightDtype,                             # W
    HendricksonLattmanDtype,                 # A
    BatchDtype,                              # B
    M_IsymDtype,                             # Y
    MTZIntDtype,                             # I
    MTZRealDtype                             # R
)
