import pytest
from pandas.testing import assert_series_equal

import reciprocalspaceship as rs


@pytest.mark.parametrize(
    "dataseries",
    [
        (rs.DataSeries(range(10), dtype=rs.PhaseDtype()), "P"),
        (rs.DataSeries(range(10), dtype=rs.HKLIndexDtype()), "H"),
        (rs.DataSeries(range(10), name="Phi", dtype=rs.PhaseDtype()), "P"),
        (rs.DataSeries(range(10), name=None), "I"),
        (rs.DataSeries(range(10), name=None, dtype=float), "R"),
        (rs.DataSeries(range(10), name="blah", dtype=float), "R"),
        (rs.DataSeries(range(10), name=None), "I"),
        (rs.DataSeries(range(10), name=None, dtype=float), "R"),
        (rs.DataSeries(["h"] * 3, name=None, dtype=object), object),
        (rs.DataSeries(["h"] * 3, name="blah", dtype=object), object),
        (rs.DataSeries(range(10), name="H"), "H"),
        (rs.DataSeries(range(10), name="K"), "H"),
        (rs.DataSeries(range(10), name="L"), "H"),
        (rs.DataSeries(range(10), name="I"), "J"),
        (rs.DataSeries(range(10), name="IMEAN"), "J"),
        (rs.DataSeries(range(10), name="SIGIMEAN"), "Q"),
        (rs.DataSeries(range(10), name="SIGI"), "Q"),
        (rs.DataSeries(range(10), name="SigI"), "Q"),
        (rs.DataSeries(range(10), name="SigF"), "Q"),
        (rs.DataSeries(range(10), name="SIGF"), "Q"),
        (rs.DataSeries(range(10), name="F"), "F"),
        (rs.DataSeries(range(10), name="F-obs"), "F"),
        (rs.DataSeries(range(10), name="ANOM"), "F"),
        (rs.DataSeries(range(10), name="PHANOM"), "P"),
        (rs.DataSeries(range(10), name="PHI"), "P"),
        (rs.DataSeries(range(10), name="PHIFMODEL"), "P"),
        (rs.DataSeries(range(10), name="F(+)"), "G"),
        (rs.DataSeries(range(10), name="F(-)"), "G"),
        (rs.DataSeries(range(10), name="SigF(+)"), "L"),
        (rs.DataSeries(range(10), name="SigF(-)"), "L"),
        (rs.DataSeries(range(10), name="I(+)"), "K"),
        (rs.DataSeries(range(10), name="I(-)"), "K"),
        (rs.DataSeries(range(10), name="SigI(+)"), "M"),
        (rs.DataSeries(range(10), name="SigI(-)"), "M"),
        (rs.DataSeries(range(10), name="HLA"), "A"),
        (rs.DataSeries(range(10), name="HLB"), "A"),
        (rs.DataSeries(range(10), name="HLC"), "A"),
        (rs.DataSeries(range(10), name="HLD"), "A"),
        (rs.DataSeries(range(10), name="M/ISYM"), "Y"),
        (rs.DataSeries(range(10), name="E"), "E"),
        (rs.DataSeries(range(10), name="batch"), "B"),
        (rs.DataSeries(range(10), name="image"), "B"),
        (rs.DataSeries(range(10), name="weight"), "W"),
        (rs.DataSeries(range(10), name="weights"), "W"),
        (rs.DataSeries(range(10), name="W"), "W"),
        (rs.DataSeries(range(10), name="FreeR_flag"), "I"),
    ],
)
def test_inference(dataseries):
    """Test DataSeries.infer_mtz_dtype()"""
    result = dataseries[0].infer_mtz_dtype()
    expected = dataseries[0].astype(dataseries[1])
    assert_series_equal(result, expected)
