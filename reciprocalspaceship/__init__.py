# Version number for reciprocalspaceship
def getVersionNumber():
    import pkg_resources

    version = pkg_resources.require("reciprocalspaceship")[0].version
    return version


__version__ = getVersionNumber()

# Top-Level API

# Import algorithms submodule

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

# fmt: on
__all__ = [
    "HKLIndexDtype",
    "IntensityDtype",
    "StructureFactorAmplitudeDtype",
    "AnomalousDifferenceDtype",
    "StandardDeviationDtype",
    "FriedelStructureFactorAmplitudeDtype",
    "StandardDeviationFriedelSFDtype",
    "FriedelIntensityDtype",
    "StandardDeviationFriedelIDtype",
    "NormalizedStructureFactorAmplitudeDtype",
    "PhaseDtype",
    "WeightDtype",
    "HendricksonLattmanDtype",
    "BatchDtype",
    "M_IsymDtype",
    "MTZIntDtype",
    "MTZRealDtype",
]
