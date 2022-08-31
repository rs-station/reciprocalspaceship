import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("sample_rate", [0.5, 1.0, 2.5, 3.0, 5.7])
@pytest.mark.parametrize("dmin", [10.0, 7.0, 5.0, 3.0, 2.0, None])
def test_get_reciprocal_grid_size(mtz_by_spacegroup, sample_rate, dmin):
    """
    Test rs.utils.get_reciprocal_grid_size() with fmodel data
    """
    dataset = rs.read_mtz(mtz_by_spacegroup)

    if dmin is None:
        dmin = dataset.compute_dHKL().dHKL.min()

    if sample_rate < 1.0:
        with pytest.raises(ValueError):
            rs.utils.get_reciprocal_grid_size(
                dataset.cell, dmin=dmin, sample_rate=sample_rate
            )

    else:
        result = rs.utils.get_reciprocal_grid_size(
            dataset.cell, dmin=dmin, sample_rate=sample_rate
        )

        # shape of output
        assert isinstance(result, list)
        assert len(result) == 3

        # real-space grid spacing must be higher resolution than dmin/sample_rate
        abc = np.array(dataset.cell.parameters[:3])
        min_size = abc / (dmin / sample_rate)
        assert (np.array(result) >= min_size).all()

        # When invoked with spacegroup symmetry-constraints, the returned grid should be
        # at least as large as without those constraints
        result_sg = rs.utils.get_reciprocal_grid_size(
            dataset.cell,
            dmin=dmin,
            sample_rate=sample_rate,
            spacegroup=dataset.spacegroup,
        )
        assert result_sg >= result
