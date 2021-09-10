import unittest

import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.fixture(
    params=[
        50.0,
        -50.0,
        250.0,
        -250.0,
        np.array([50.0, -50.0, 250.0, -250.0]),
        rs.DataSeries([50.0, -50.0, 250.0, -250.0], dtype="Phase"),
    ]
)
def phase_deg(request):
    """Yields phases (in degrees) for testing"""
    return request.param


@pytest.mark.parametrize("deg", [True, False])
def test_canonicalize_phases(phase_deg, deg):
    # Test canonicalize_phases

    expected_phase = ((phase_deg + 180.0) % 360.0) - 180.0

    if not deg:
        phase_deg = np.deg2rad(phase_deg)
        expected_phase = np.deg2rad(expected_phase)
    p = rs.utils.canonicalize_phases(phase_deg, deg)
    if isinstance(p, rs.DataSeries):
        p = p.to_numpy(np.float32)
        expected_phase = expected_phase.to_numpy(np.float32)

    assert np.allclose(p, expected_phase)


@pytest.mark.parametrize("deg", [True, 1, 1.0, object, None])
def test_canonicalize_phases_typeerror(deg):
    # canonicalize_phases raises TypeError if deg is not boolean

    if deg == True or deg == False:
        p = rs.utils.canonicalize_phases(1.0, deg=deg)
        assert p == 1.0
    else:
        with pytest.raises(TypeError):
            p = rs.utils.canonicalize_phases(0.0, deg=deg)
