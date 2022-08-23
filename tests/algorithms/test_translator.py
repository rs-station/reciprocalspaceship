import numpy as np
import pytest

from reciprocalspaceship.algorithms import translator


@pytest.mark.parametrize(
    "translator_class", [translator.PolarTranslator, translator.NonPolarTranslator]
)
def test_initialization(hkls, translator_class):
    """
    Test initialization of PolarTranslator and NonPolarTranslator
    """
    phi_ref = np.linspace(0, 360, len(hkls))
    phi = np.linspace(0, 360, len(hkls))
    dhkl = np.linspace(1, 100, len(hkls))

    # Malformed Miller index array should raise ValueError
    with pytest.raises(ValueError):
        solver = translator_class(hkls[:, :2], phi_ref, phi, dhkl)

    # Mismatched array lengths should cause ValueErrors
    with pytest.raises(ValueError):
        solver = translator_class(hkls[:-1], phi_ref, phi, dhkl)
    with pytest.raises(ValueError):
        solver = translator_class(hkls, phi_ref[:-1], phi, dhkl)
    with pytest.raises(ValueError):
        solver = translator_class(hkls, phi_ref, phi[:-1], dhkl)
    with pytest.raises(ValueError):
        solver = translator_class(hkls, phi_ref, phi, dhkl[:-1])

    solver = translator_class(hkls, phi_ref, phi, dhkl)

    # Since phi_ref == phi, rmsd should be 0.0
    assert solver.phase_rmsd() == 0.0

    # Loss function with initial translation vector should be 0.0
    assert np.array_equal(solver.translation, np.array([0.0, 0.0, 0.0]))
    assert solver.evaluate(solver.translation) == 0.0

    # fit() methods should return [0.0, 0.0, 0.0] -- no translation needed
    assert np.array_equal(solver.fit(), np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "translator_class", [translator.PolarTranslator, translator.NonPolarTranslator]
)
@pytest.mark.parametrize(
    "translation",
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])],
)
def test_evaluate_integers(hkls, translator_class, translation):
    """
    Test evaluate() returns 0.0 with integer fractional cell lengths
    """
    phi_ref = np.linspace(0, 360, len(hkls))
    phi = np.linspace(0, 360, len(hkls))
    dhkl = np.linspace(1, 100, len(hkls))

    solver = translator_class(hkls, phi_ref, phi, dhkl)
    assert solver.evaluate(translation) == 0.0


@pytest.mark.parametrize(
    "translator_class", [translator.PolarTranslator, translator.NonPolarTranslator]
)
@pytest.mark.parametrize(
    "translation",
    [
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([0.25, 0.25, 0.25]),
        np.array([0.1, 0.2, 0.3]),
    ],
)
def test_evaluate_fractional(hkls, translator_class, translation):
    """
    Test evaluate() returns > 0.0 with fractional cell lengths
    """
    phi_ref = np.linspace(0, 360, len(hkls))
    phi = np.linspace(0, 360, len(hkls))
    dhkl = np.linspace(1, 100, len(hkls))

    solver = translator_class(hkls, phi_ref, phi, dhkl)
    assert solver.evaluate(translation) > 0.0
