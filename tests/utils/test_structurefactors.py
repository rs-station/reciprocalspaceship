import pytest
import numpy as np
import reciprocalspaceship as rs
import gemmi

@pytest.mark.parametrize(
    "sfs_phases", [
        (np.random.rand(10), np.random.rand(10)),
        (list(np.random.rand(10)), list(np.random.rand(10))),
        ([], []),
        (1.0, 90.),
        (rs.DataSeries(np.linspace(1, 20, 10), name="F", dtype="SFAmplitude"),
         rs.DataSeries(np.random.rand(10), name="Phi", dtype="Phase"))
    ]
)
def test_to_structurefactor(sfs_phases):
    """
    Test rs.utils.to_structurefactor() returns complex structure factors
    when given amplitudes and phases.
    """
    sfamps = sfs_phases[0]
    phases = sfs_phases[1]
    sfs = rs.utils.to_structurefactor(sfamps, phases)

    # Handle DataSeries 
    if isinstance(sfamps, rs.DataSeries):
        sfamps = sfamps.to_numpy()
    if isinstance(phases, rs.DataSeries):
        phases = phases.to_numpy()

    reference = sfamps*np.exp(1j*np.deg2rad(phases))
    assert np.iscomplexobj(sfs)
    assert np.isclose(sfs, reference).all()


def test_from_structurefactor():
    """
    Test rs.utils.from_structurefactor() returns structure factor 
    amplitudes and phases when given complex numpy array
    """
    sfs = np.random.rand(10)*np.exp(1j*np.random.rand(10))
    sf, phase = rs.utils.from_structurefactor(sfs)

    assert len(sf) == len(sfs)
    assert len(phase) == len(sfs)
    assert isinstance(sf, rs.DataSeries)
    assert isinstance(phase, rs.DataSeries)
    assert isinstance(sf.dtype, rs.StructureFactorAmplitudeDtype)
    assert isinstance(phase.dtype, rs.PhaseDtype)
    assert np.isclose(sf.to_numpy(), np.abs(sfs)).all()
    assert np.isclose(phase.to_numpy(), np.angle(sfs, deg=True)).all()

@pytest.mark.parametrize(
    "sg", [
        gemmi.SpaceGroup(1),
        gemmi.SpaceGroup(1).operations(),
        "invalid",
        None,
    ]
)
def test_structurefactor_multiplicity_valueerror(sg):
    """
    Test rs.utils.compute_structurefactor_multiplicity() raises 
    ValueError when invoked with invalid spacegroup
    """
    H  = np.array([[1, 1, 1]])
    if isinstance(sg, gemmi.SpaceGroup) or isinstance(sg, gemmi.GroupOps):
        epsilon = rs.utils.compute_structurefactor_multiplicity(H, sg)
    else:
        with pytest.raises(ValueError):
            epsilon = rs.utils.compute_structurefactor_multiplicity(H, sg)
