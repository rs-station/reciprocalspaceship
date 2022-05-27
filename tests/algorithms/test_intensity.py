import numpy as np
import pytest

import reciprocalspaceship as rs


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("output_columns", [None, 
                                            ("I_backcalc", "SigI_backcalc")]
                         )
def test_compute_intensity_from_structurefactor(ref_hewl, inplace, output_columns):
    """
    Test rs.algorithms.compute_intensity_from_structurefactor() returns
    intensities and intensity error estimates as expected based on assumptions
    """

    result = rs.algorithms.compute_intensity_from_structurefactor(
        ref_hewl, 
        "F",
        "SIGF",
        output_columns=output_columns,
        inplace=inplace
    )
    
    if inplace:
        assert id(result) == id(ref_hewl)
    else:
        assert id(result) != id(ref_hewl)
     
    defaults = ("I_back", "SigI_back")
    if output_columns:
        o1, o2 = output_columns
    else:
        o1, o2 = defaults
    
    # confirm types of new columns
    assert isinstance(result[o1].dtype, rs.IntensityDtype)
    assert isinstance(result[o2].dtype, rs.StandardDeviationDtype)
    
    # confirm arithmetic
    F = result['F'].to_numpy()
    SigF = result['SIGF'].to_numpy()
    I = result[o1].to_numpy()
    SigI = result[o2].to_numpy()
    assert np.isclose(I, SigF * SigF + F * F).all()
    assert np.isclose(SigI, np.abs(2 * F * SigF)).all()

def test_compute_intensity_from_structurefactor_failure(ref_hewl):
    """
    Test that rs.algorithms.compute_intensity_from_structurefactor() throws a
    ValueError rather than overwriting columns
    """
    
    with pytest.raises(ValueError):
        rs.algorithms.compute_intensity_from_structurefactor(
            ref_hewl, 
            'F', 
            'SIGF',
            output_columns=('I', 'SIGI'))