import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize(
    "F_SigF",
    [
        (np.random.rand(10), np.random.rand(10)),
        (list(np.random.rand(10)), list(np.random.rand(10))),
        ([], []),
        (50, 5),
        (
            rs.DataSeries(np.random.rand(10), name="F", dtype="SFAmplitude"),
            rs.DataSeries(np.random.rand(10), name="SigF", dtype="Stddev"),
        ),
    ],
)
def test_compute_intensity_from_structurefactor(F_SigF):
    """
    Test rs.algorithms.compute_intensity_from_structurefactor() returns
    intensities and intensity error estimates as expected based on assumptions
    """

    F = F_SigF[0]
    SigF = F_SigF[1]

    I, SigI = rs.algorithms.compute_intensity_from_structurefactor(F, SigF)

    assert np.isclose(I, SigF * SigF + F * F)
    assert np.isclose(SigI, np.abs(2 * F * SigF))
