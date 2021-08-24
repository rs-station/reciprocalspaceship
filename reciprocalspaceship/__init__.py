# Version number for reciprocalspaceship
def getVersionNumber():
    import pkg_resources

    version = pkg_resources.require("reciprocalspaceship")[0].version
    return version


__version__ = getVersionNumber()

# Import algorithms submodule
from reciprocalspaceship import algorithms

from .concat import concat
from .dataseries import DataSeries

# Top-Level API
from .dataset import DataSet

# Add support for MTZ data types:
# see http://www.ccp4.ac.uk/html/f2mtz.html
# fmt: off
from .dtypes import AnomalousDifferenceDtype  # D
from .dtypes import BatchDtype  # B
from .dtypes import FriedelIntensityDtype  # K
from .dtypes import FriedelStructureFactorAmplitudeDtype  # G
from .dtypes import HendricksonLattmanDtype  # A
from .dtypes import HKLIndexDtype  # H
from .dtypes import IntensityDtype  # J
from .dtypes import M_IsymDtype  # Y
from .dtypes import MTZIntDtype  # I
from .dtypes import MTZRealDtype  # R
from .dtypes import NormalizedStructureFactorAmplitudeDtype  # E
from .dtypes import PhaseDtype  # P
from .dtypes import StandardDeviationDtype  # Q
from .dtypes import StandardDeviationFriedelIDtype  # M
from .dtypes import StandardDeviationFriedelSFDtype  # L
from .dtypes import StructureFactorAmplitudeDtype  # F
from .dtypes import WeightDtype  # W
from .dtypes import summarize_mtz_dtypes
from .io import read_crystfel, read_csv, read_mtz, read_pickle, read_precognition

# Import algorithms submodule


# fmt: on
