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
    result = rs.stats.compute_completeness(
        data_hewl, bins=bins, anomalous=anomalous, dmin=dmin
    )

    assert len(result) == bins + 1
    assert result.index[-1] == "overall"
    assert float(result.index[-2].split()[-1]) == dmin
    assert (result["completeness"] <= 1.0).all()


@pytest.mark.parametrize("bins", [1, 5, 10, 20])
@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0])
def test_compute_completeness_valueerror(data_merged, bins, anomalous, dmin):
    """
    Test rs.stats.compute_completeness() raises ValueError if given
    merged data outside of the reciprocal ASU. Behavior should not depend
    on values of additional arguments
    """
    data = data_merged.stack_anomalous()

    with pytest.raises(ValueError):
        result = rs.stats.compute_completeness(
            data, bins=bins, anomalous=anomalous, dmin=dmin
        )


@pytest.mark.parametrize("sg", [1, 4, 5, 19, 96])
@pytest.mark.parametrize("bins", [1, 5, 10])
@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0])
@pytest.mark.parametrize("merged", [True, False])
def test_compute_completeness_full(sg, bins, anomalous, dmin, merged):
    """
    Test rs.stats.compute_completeness with synthetic data that is 100% complete
    """
    cell = [30, 30, 30, 90, 90, 90]
    H = rs.utils.generate_reciprocal_asu(cell, sg, dmin, anomalous)
    data = rs.DataSet(
        {"H": H[:, 0], "K": H[:, 1], "L": H[:, 2]},
        cell=cell,
        spacegroup=sg,
        merged=merged,
    )
    data["I"] = 1.0
    data.infer_mtz_dtypes(inplace=True)
    data.set_index(["H", "K", "L"], inplace=True)

    # Merged, anomalous data is only supported in two-column format
    if anomalous and merged:
        data = data.unstack_anomalous()

    result = rs.stats.compute_completeness(
        data, bins=bins, anomalous=anomalous, dmin=dmin
    )
    assert (result["completeness"] == 1.0).all()


@pytest.mark.parametrize("bins", [1, 5, 10])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0])
def test_compute_completeness_half_anomalous(bins, dmin):
    """
    Test rs.stats.compute_completeness with synthetic data that is half complete.

    This function generates reciprocal ASU reflections in one Friedel half, but computes
    anomalous completeness, using P1 data to avoid centrics
    """
    cell = [30, 30, 30, 90, 90, 90]
    sg = "P 1"
    H = rs.utils.generate_reciprocal_asu(cell, sg, dmin, anomalous=False)
    data = rs.DataSet(
        {"H": H[:, 0], "K": H[:, 1], "L": H[:, 2]},
        cell=cell,
        spacegroup=sg,
        merged=False,
    )
    data["I"] = 1.0
    data.infer_mtz_dtypes(inplace=True)
    data.set_index(["H", "K", "L"], inplace=True)

    result = rs.stats.compute_completeness(data, bins=bins, anomalous=True, dmin=dmin)
    assert (result["completeness"] == 0.5).all()
