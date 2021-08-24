# ExtensionDtypes are appended to the end of the Dtype registry.
# Since we want to overwrite a few of the one-letter strings, we need
# to make sure that rs ExtensionDtypes appear first in the registry.
# This will be handled by reversing the list.
try:
    from pandas.core.dtypes.base import registry
except:
    from pandas.core.dtypes.base import _registry as registry
registry.dtypes = registry.dtypes[::-1]
