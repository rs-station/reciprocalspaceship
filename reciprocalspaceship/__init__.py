# Version number for reciprocalspaceship
def getVersionNumber():
    import pkg_resources

    version = pkg_resources.require("reciprocalspaceship")[0].version
    return version


__version__ = getVersionNumber()

# Import algorithms submodule
from reciprocalspaceship import algorithms

# Top-Level API
from reciprocalspaceship.concat import concat
from reciprocalspaceship.dataseries import DataSeries
from reciprocalspaceship.dataset import DataSet

# Add support for MTZ data types:
# see http://www.ccp4.ac.uk/html/f2mtz.html
from reciprocalspaceship.dtypes import AnomalousDifferenceDtype  # D
from reciprocalspaceship.dtypes import BatchDtype  # B
from reciprocalspaceship.dtypes import FriedelIntensityDtype  # K
from reciprocalspaceship.dtypes import FriedelStructureFactorAmplitudeDtype  # G
from reciprocalspaceship.dtypes import HendricksonLattmanDtype  # A
from reciprocalspaceship.dtypes import HKLIndexDtype  # H
from reciprocalspaceship.dtypes import IntensityDtype  # J
from reciprocalspaceship.dtypes import M_IsymDtype  # Y
from reciprocalspaceship.dtypes import MTZIntDtype  # I
from reciprocalspaceship.dtypes import MTZRealDtype  # R
from reciprocalspaceship.dtypes import NormalizedStructureFactorAmplitudeDtype  # E
from reciprocalspaceship.dtypes import PhaseDtype  # P
from reciprocalspaceship.dtypes import StandardDeviationDtype  # Q
from reciprocalspaceship.dtypes import StandardDeviationFriedelIDtype  # M
from reciprocalspaceship.dtypes import StandardDeviationFriedelSFDtype  # L
from reciprocalspaceship.dtypes import StructureFactorAmplitudeDtype  # F
from reciprocalspaceship.dtypes import WeightDtype  # W
from reciprocalspaceship.dtypes import summarize_mtz_dtypes
from reciprocalspaceship.io import (
    read_crystfel,
    read_csv,
    read_mtz,
    read_pickle,
    read_precognition,
)
