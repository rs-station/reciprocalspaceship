from .anomalousdifference import AnomalousDifferenceDtype
from .batch import BatchDtype
from .hklindex import HKLIndexDtype
from .intensity import FriedelIntensityDtype, IntensityDtype
from .m_isym import M_IsymDtype
from .mtzint import MTZIntDtype
from .mtzreal import MTZRealDtype
from .phase import HendricksonLattmanDtype, PhaseDtype
from .stddev import (
    StandardDeviationDtype,
    StandardDeviationFriedelIDtype,
    StandardDeviationFriedelSFDtype,
)
from .structurefactor import (
    FriedelStructureFactorAmplitudeDtype,
    NormalizedStructureFactorAmplitudeDtype,
    StructureFactorAmplitudeDtype,
)
from .weight import WeightDtype

# ExtensionDtypes are appended to the end of the Dtype registry.
# Since we want to overwrite a few of the one-letter strings, we need
# to make sure that rs ExtensionDtypes appear first in the registry.
# This will be handled by reversing the list.
try:
    from pandas.core.dtypes.base import registry
except:
    from pandas.core.dtypes.base import _registry as registry
registry.dtypes = registry.dtypes[::-1]

from .summarize import summarize_mtz_dtypes
