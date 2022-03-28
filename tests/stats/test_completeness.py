import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("bins", [1, 5, 10, 20])
@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0, None])
def test_compute_completeness_hewl(data_hewl, bins, anomalous, dmin):
    """
    Test rs.stats.compute_completeness() with experimental HEWL data,
    both merged and unmerged.

    This test does not assess numerical values, but just that the output
    behaves as expected for given arguments.
    """
    result = rs.stats.compute_completeness(
        data_hewl, bins=bins, anomalous=anomalous, dmin=dmin, unobserved_value=0
    )

    assert len(result) == bins + 1
    assert result.index[-1] == "overall"
    assert (result.to_numpy() <= 1.0).all()

    if dmin is None:
        dmin = data_hewl.compute_dHKL()["dHKL"].min()

    # Label includes two decimal places
    assert np.isclose(float(result.index[-2].split()[-1]), dmin, atol=0.01)

    if anomalous:
        assert len(result.columns) == 3
    else:
        assert len(result.columns) == 1


@pytest.mark.parametrize("bins", [1, 5, 10, 20])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0, None])
def test_compute_completeness_auto(data_hewl, bins, dmin):
    """Test rs.stats.compute_completeness() in 'auto' anomalous mode"""
    if not data_hewl.merged:
        result = rs.stats.compute_completeness(
            data_hewl, bins=bins, dmin=dmin, unobserved_value=0
        )
        assert len(result.columns) == 3
    elif data_hewl.merged:
        # Test merged with anomalous columns
        data = data_hewl[["I(+)", "SIGI(+)", "I(-)", "SIGI(-)"]]
        result = rs.stats.compute_completeness(
            data, bins=bins, dmin=dmin, unobserved_value=0
        )
        assert len(result.columns) == 3

        # Test merged with non-anomalous columns
        data = data_hewl[["IMEAN", "SIGIMEAN"]]
        result = rs.stats.compute_completeness(
            data, bins=bins, dmin=dmin, unobserved_value=0
        )
        assert len(result.columns) == 1
    # Test AttributeError
    with pytest.raises(AttributeError):
        data_hewl.merged = None
        result = rs.stats.compute_completeness(
            data_hewl, bins=bins, dmin=dmin, unobserved_value=0
        )


@pytest.mark.parametrize("bins", [1, 5, 10, 20])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0, None])
def test_compute_completeness_nonanomalous(data_hewl, bins, dmin):
    """
    Test rs.stats.compute_completeness() with experimental HEWL data,
    both merged and unmerged.

    This test confirms that the non-anomalous completeness (after merging
    Friedel pairs) is equivalent whether called with `anomalous=True` or
    `anomalous=False`
    """
    result_nonanom = rs.stats.compute_completeness(
        data_hewl, bins=bins, anomalous=False, dmin=dmin, unobserved_value=0
    )
    result_anom = rs.stats.compute_completeness(
        data_hewl, bins=bins, anomalous=True, dmin=dmin, unobserved_value=0
    )
    assert np.array_equal(
        result_nonanom[("completeness", "non-anomalous")],
        result_anom[("completeness", "non-anomalous")],
    )


@pytest.mark.parametrize("bins", [1, 5, 10, 20])
@pytest.mark.parametrize("anomalous", [True, False, "auto"])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0, None])
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
    assert (result["completeness"].to_numpy() == 1.0).all()


@pytest.mark.parametrize("bins", [1, 5, 10])
@pytest.mark.parametrize("dmin", [5.0, 4.0, 3.0])
@pytest.mark.parametrize("anomalous", [True, False])
def test_compute_completeness_half_anomalous(bins, dmin, anomalous):
    """
    Test rs.stats.compute_completeness with synthetic data that is half complete.

    This function generates reciprocal ASU reflections in one Friedel half, then
    computes completeness.
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

    result = rs.stats.compute_completeness(
        data, bins=bins, anomalous=anomalous, dmin=dmin
    )
    if anomalous:
        assert (result[("completeness", "non-anomalous")] == 1.0).all()
        assert (result[("completeness", "all")] == 0.5).all()
        assert (result[("completeness", "anomalous")] == 0.5).all()
    else:
        assert (result[("completeness", "non-anomalous")] == 1.0).all()
