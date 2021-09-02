import numpy as np
import pytest

from reciprocalspaceship.utils import angle_between


def normalize(vec):
    return vec / np.linalg.norm(vec, axis=-1)[..., None]


@pytest.mark.parametrize("deg", [True, False])
def test_angle_between(deg):
    # First test equivalence with a naive formula on an easy set of vectors
    # Incidentally this also tests some pretty intense broadcasting
    v2 = np.mgrid[
        -1.05:1.15:0.1,
        -1.05:1.15:0.1,
        -1.05:1.15:0.1,
    ].T
    v1 = np.array([0.0, 0.0, 1.0])
    angle = angle_between(v1, v2, deg=deg)

    naive_cos = (normalize(v1) * normalize(v2)).sum(-1)
    naive_angle = np.arccos(naive_cos)
    if deg:
        naive_angle = np.rad2deg(naive_angle)

    assert np.allclose(naive_angle, angle)

    # Now we will just test numerical stability by making some vectors that are
    # very close together.
    x = 10 ** np.mgrid[-1:-301:-1.0]
    v2 = np.column_stack((x, np.zeros_like(x), np.zeros_like(x))) + v1

    angle = angle_between(v1, v2, deg=deg)
    assert np.all(np.isfinite(angle))

    # Finally just test that exactly equal vectors don't lead to any instability
    angle = angle_between(v2, v2, deg=deg)
    assert np.all(angle == 0.0)
