import numpy as np

from reciprocalspaceship.decorators import spacegroupify


def apply_to_hkl(H, op):
    """
    Apply symmetry operator to hkls.

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    op : gemmi.Op
        gemmi symmetry operator to be applied

    Returns
    -------
    result : np.ndarray(int32)
        n x 3 array of Miller indices after operator application

    Raises
    ------
    RuntimeError
        If `op` generates fractional Miller indices when applied to `H`
    """
    # Case 1: No risk of fractional Miller indices
    if ((np.array(op.rot) / op.DEN) % 1 == 0).all():
        return np.floor_divide(np.matmul(H, op.rot), op.DEN).astype("int32")

    # Case 2: Depends on input Miller indices
    else:
        Hnew = np.divide(np.matmul(H, op.rot), op.DEN)
        # Check for issues
        if np.any(np.mod(Hnew, 1)):
            raise RuntimeError(
                f"Applying {op} to Miller indices produced non-integer results. "
                f"Fractional Miller indices are not currently supported."
            )
        else:
            return Hnew.astype("int32")


def phase_shift(H, op):
    """
    Calculate phase shift for symmetry operator.

    Parameters
    ----------
    H : array
        n x 3 array of Miller indices
    op : gemmi.Op
        gemmi symmetry operator to be applied

    Returns
    -------
    result : array
        array of phase shifts
    """
    return -2 * np.pi * np.matmul(H, op.tran) / op.DEN


@spacegroupify
def is_polar(spacegroup):
    """
    Classify whether spacegroup is polar

    Parameters
    ----------
    spacegroup : str, int, gemmi.SpaceGroup
        Spacegroup to classify as polar

    Returns
    -------
    bool
        Whether the spacegroup is polar
    """
    sym_ops = spacegroup.operations().sym_ops
    a = np.array([op.rot for op in sym_ops])
    return ~(a < 0).any(axis=1).any(axis=0).all()


@spacegroupify
def polar_axes(spacegroup):
    """
    Classify which cell axes are polar in given spacegroup

    Notes
    -----
    - This method assumes that the lattice basis vectors are a, b, and c.
      Specifically, trigonal spacegroups with rhombohedral centering are not
      supported.

    Parameters
    ----------
    spacegroup : str, int, gemmi.SpaceGroup
        Spacegroup to classify axes as polar

    Returns
    -------
    List[bool]
        Whether a-, b-, or c-axis is polar

    Raises
    ------
    ValueError
        If trigonal spacegroup is provided with rhombohedral Bravais lattice
    """
    if spacegroup.crystal_system_str() == "trigonal" and spacegroup.xhm().endswith(
        ":R"
    ):
        raise ValueError(
            f"Spacegroup {spacegroup} is not supported. Please provide with hexagonal Bravais lattice"
        )

    sym_ops = spacegroup.operations().sym_ops
    a = np.array([op.rot for op in sym_ops])
    return (~(a < 0).any(axis=1).any(axis=0)).tolist()
