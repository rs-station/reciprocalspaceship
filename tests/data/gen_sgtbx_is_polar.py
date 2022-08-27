"""
Generate test data using sgtbx classifications of polar spacegroups
"""

import pandas as pd
from cctbx import sgtbx

# Classify all space group settings
is_polar = []
a_polars = []
b_polars = []
c_polars = []
xhms = []
for sg in sgtbx.space_group_symbol_iterator():
    sginfo = sgtbx.space_group(sg).info()
    xhm = sg.universal_hermann_mauguin()
    sgtbx_polar = sginfo.number_of_continuous_allowed_origin_shifts() > 0
    a_polar = sginfo.is_allowed_origin_shift([0.1, 0, 0], tolerance=1e-3)
    b_polar = sginfo.is_allowed_origin_shift([0, 0.1, 0], tolerance=1e-3)
    c_polar = sginfo.is_allowed_origin_shift([0, 0, 0.1], tolerance=1e-3)

    xhms.append(xhm)
    is_polar.append(sgtbx_polar)
    a_polars.append(a_polar)
    b_polars.append(b_polar)
    c_polars.append(c_polar)

# Write out CSV
df = pd.DataFrame(
    {
        "xhm": xhms,
        "is_polar": is_polar,
        "is_a_polar": a_polars,
        "is_b_polar": b_polars,
        "is_c_polar": c_polars,
    }
)
df.to_csv("sgtbx/sgtbx_polar.csv", index=None)
