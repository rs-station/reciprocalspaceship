"""
Generate test data using sgtbx classifications of polar spacegroups
"""

import pandas as pd
from cctbx import sgtbx

# Classify all space group settings
is_polar = []
xhms = []
for sg in sgtbx.space_group_symbol_iterator():
    sgtbx_polar = (
        sgtbx.space_group(sg).info().number_of_continuous_allowed_origin_shifts() > 0
    )
    xhm = sg.universal_hermann_mauguin()
    is_polar.append(sgtbx_polar)
    xhms.append(xhm)

# Write out CSV
df = pd.DataFrame({"xhm": xhms, "is_polar": is_polar})
df.to_csv("sgtbx/sgtbx_polar.csv", index=None)
