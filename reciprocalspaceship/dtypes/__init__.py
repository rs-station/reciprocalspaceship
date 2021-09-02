from reciprocalspaceship.dtypes.anomalousdifference import AnomalousDifferenceDtype
from reciprocalspaceship.dtypes.batch import BatchDtype
from reciprocalspaceship.dtypes.hklindex import HKLIndexDtype
from reciprocalspaceship.dtypes.intensity import FriedelIntensityDtype, IntensityDtype
from reciprocalspaceship.dtypes.m_isym import M_IsymDtype
from reciprocalspaceship.dtypes.mtzint import MTZIntDtype
from reciprocalspaceship.dtypes.mtzreal import MTZRealDtype
from reciprocalspaceship.dtypes.phase import HendricksonLattmanDtype, PhaseDtype
from reciprocalspaceship.dtypes.stddev import (
    StandardDeviationDtype,
    StandardDeviationFriedelIDtype,
    StandardDeviationFriedelSFDtype,
)
from reciprocalspaceship.dtypes.structurefactor import (
    FriedelStructureFactorAmplitudeDtype,
    NormalizedStructureFactorAmplitudeDtype,
    StructureFactorAmplitudeDtype,
)
from reciprocalspaceship.dtypes.weight import WeightDtype

# ExtensionDtypes are appended to the end of the Dtype registry.
# Since we want to overwrite a few of the one-letter strings, we need
# to make sure that rs ExtensionDtypes appear first in the registry.
# This will be handled by reversing the list.
try:
    from pandas.core.dtypes.base import registry
except:
    from pandas.core.dtypes.base import _registry as registry
registry.dtypes = registry.dtypes[::-1]

from reciprocalspaceship.dtypes.summarize import summarize_mtz_dtypes
