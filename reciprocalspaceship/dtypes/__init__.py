from .hklindex import HKLIndexDtype
from .mtzint import MTZIntDtype
from .mtzreal import MTZRealDtype
from .intensity import (
    IntensityDtype,
    IntensityFriedelDtype
)
from .structurefactor import (
    StructureFactorAmplitudeDtype,
    StructureFactorAmplitudeFriedelDtype,
    ScaledStructureFactorAmplitudeDtype
)
from .anomalousdifference import AnomalousDifferenceDtype
from .stddev import (
    StandardDeviationDtype,
    StandardDeviationSFFriedelDtype,
    StandardDeviationIFriedelDtype
)
from .phase import PhaseDtype, HendricksonLattmanDtype
from .weight import WeightDtype
from .batch import BatchDtype
from .m_isym import M_IsymDtype

# ExtensionDtypes are appended to the end of the Dtype registry.
# Since we want to overwrite a few of the one-letter strings, we need
# to make sure that rs ExtensionDtypes appear first in the registry.
# This will be handled by reversing the list.
from pandas.core.dtypes.dtypes import registry
registry.dtypes = registry.dtypes[::-1]
