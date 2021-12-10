import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("bins", [1, 5, 10, 20])
@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0])
def test_compute_completeness_hewl(data_hewl, bins, anomalous, dmin):
    """
    Test rs.stats.compute_completeness() with experimental HEWL data,
    both merged and unmerged..

    This test does not assess numerical values, but just that the output
    behaves as expected for given arguments.
    """
    result = rs.stats.compute_completeness(data_hewl, bins=bins, dmin=dmin)

    assert len(result) == bins + 1
    assert result.index[-1] == "overall"
    assert float(result.index[-2].split()[-1]) == dmin
    assert (result["completeness"] <= 1.0).all()
