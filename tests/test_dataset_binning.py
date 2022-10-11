import numpy as np
import pytest


@pytest.mark.parametrize("bins", [5, 10, 20, 50, [100.0, 1.0]])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("return_labels", [True, False])
@pytest.mark.parametrize("return_edges", [True, False])
def test_assign_resolution_bins(
    data_fmodel, bins, inplace, return_labels, return_edges
):
    """Test DataSet.assign_resolution_bins() arguments"""

    result = data_fmodel.assign_resolution_bins(
        bins=bins,
        inplace=inplace,
        return_labels=return_labels,
        return_edges=return_edges,
    )

    if return_labels and return_edges:
        result, labels, edges = result
    elif return_labels:
        result, labels = result
    elif return_edges:
        result, edges = result

    # Test bins
    assert "bin" in result.columns
    if isinstance(bins, int):
        assert len(result["bin"].unique()) == bins
        assert result.bin.max() == bins - 1
    else:
        assert all(result["bin"].to_numpy() == 0)

    # Test inplace
    if inplace:
        assert id(result) == id(data_fmodel)
    else:
        assert id(result) != id(data_fmodel)

    # Test labels
    if return_labels:
        assert isinstance(labels, list)
        assert all([isinstance(l, str) for l in labels])
        if isinstance(bins, int):
            assert len(labels) == bins
        else:
            assert len(labels) == len(bins) - 1

    # Test edges
    if return_edges:
        assert np.all(np.diff(edges) < 0)
        assert isinstance(edges, np.ndarray)
        if isinstance(bins, int):
            assert len(edges) == bins + 1
        else:
            assert all(edges == bins)


@pytest.mark.parametrize("bins", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("reverse", [True, False])
def test_binedges_equivalence(data_fmodel, bins, reverse):
    """
    Test DataSet.assign_resolution_bins() generates consistent assignment when
    provided bin edges from `return_edges=True`. Tests using bin edges provided
    in both ascending or descending order
    """

    expected, edges = data_fmodel.assign_resolution_bins(
        bins, return_labels=False, return_edges=True
    )

    # Reversed bin edges should reverse assignments
    if reverse:
        result = data_fmodel.assign_resolution_bins(
            edges[::-1], return_labels=False, return_edges=False
        )
        assert all(expected["bin"] == ((bins - 1) - result["bin"]))

    # All assignments should be equivalent
    else:
        result = data_fmodel.assign_resolution_bins(
            edges, return_labels=False, return_edges=False
        )
        assert all(expected["bin"] == result["bin"])


@pytest.mark.parametrize("bins", [[8.0, 12.0], [12.0, 8.0], [12.0, 20.0]])
@pytest.mark.parametrize("inplace", [True, False])
def test_bindges_drops_reflections(data_fmodel, bins, inplace):
    """
    Test DataSet.assign_resolution_bins() drops reflections outside of
    resolution range when provided explicit bin edges
    """
    nrows = len(data_fmodel)
    result = data_fmodel.assign_resolution_bins(
        bins, inplace=inplace, return_labels=False, return_edges=False
    )

    assert all(result["bin"] == 0)

    dHKL = result.compute_dHKL()["dHKL"]
    assert all(dHKL <= max(bins))
    assert all(dHKL >= min(bins))
    assert len(result) < nrows

    # Test inplace
    if inplace:
        assert id(result) == id(data_fmodel)
        assert len(data_fmodel) == len(result)
    else:
        assert id(result) != id(data_fmodel)
        assert len(data_fmodel) == nrows
