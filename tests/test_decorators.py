import gemmi
import numpy as np
import pytest

from reciprocalspaceship.decorators import cellify, spacegroupify


@spacegroupify
def _function_bare(spacegroup=None, arg2=None):
    """spacegroup argument should be coerced to gemmi.SpaceGroup"""
    return spacegroup, arg2


@spacegroupify()
def _function_empty(spacegroup=None, arg2=None):
    """spacegroup argument should be coerced to gemmi.SpaceGroup"""
    return spacegroup, arg2


@spacegroupify("arg2")
def _function_arg(spacegroup=None, arg2=None):
    """arg2 argument should be coerced to gemmi.SpaceGroup"""
    return spacegroup, arg2


@pytest.mark.parametrize("function", [_function_bare, _function_empty])
@pytest.mark.parametrize("sg", [None, 1, gemmi.SpaceGroup(1), "P 1", 1.0])
@pytest.mark.parametrize("arg2", [1, 2, 3])
def test_spacegroupify_default(function, sg, arg2):
    """Test spacegroupify decorator with default arguments"""

    if isinstance(sg, float):
        with pytest.raises(ValueError):
            result_spacegroup, result_arg2 = function(sg, arg2=arg2)
    else:
        result_spacegroup, result_arg2 = function(sg, arg2=arg2)

        # arg2 argument should be unchanged
        assert result_arg2 == arg2

        # check spacegroup coercion to gemmi instance
        if sg is None:
            assert result_spacegroup is None
        elif isinstance(sg, gemmi.SpaceGroup):
            assert result_spacegroup.xhm() == sg.xhm()
        else:
            assert isinstance(result_spacegroup, gemmi.SpaceGroup)
            expected = gemmi.SpaceGroup(sg)
            assert result_spacegroup.xhm() == expected.xhm()


@pytest.mark.parametrize("sg", [1, 2, 3])
@pytest.mark.parametrize("arg2", [None, 1, gemmi.SpaceGroup(1), "P 1", 1.0])
def test_spacegroupify_arg(sg, arg2):
    """Test spacegroupify decorator with provided arguments"""

    if isinstance(arg2, float):
        with pytest.raises(ValueError):
            result_spacegroup, result_arg2 = _function_arg(sg, arg2=arg2)
    else:
        result_spacegroup, result_arg2 = _function_arg(sg, arg2=arg2)

        # spacegroup argument should be unchanged
        assert result_spacegroup == sg

        # check arg2 coercion to gemmi instance
        if arg2 is None:
            assert result_arg2 is None
        elif isinstance(arg2, gemmi.SpaceGroup):
            assert result_arg2.xhm() == arg2.xhm()
        else:
            assert isinstance(result_arg2, gemmi.SpaceGroup)
            expected = gemmi.SpaceGroup(arg2)
            assert result_arg2.xhm() == expected.xhm()


@cellify
def _function2_bare(cell=None, arg2=None):
    """cell argument should be coerced to gemmi.UnitCell"""
    return cell, arg2


@cellify()
def _function2_empty(cell=None, arg2=None):
    """cell argument should be coerced to gemmi.UnitCell"""
    return cell, arg2


@cellify("arg2")
def _function2_arg(cell=None, arg2=None):
    """arg2 argument should be coerced to gemmi.UnitCell"""
    return cell, arg2


@pytest.mark.parametrize("function", [_function2_bare, _function2_empty])
@pytest.mark.parametrize(
    "cell",
    [
        None,
        (1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
        [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
        np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
        gemmi.UnitCell(1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
        1.0,
    ],
)
@pytest.mark.parametrize("arg2", [1, 2, 3])
def test_cellify_default(function, cell, arg2):
    """Test cellify decorator with default arguments"""

    if isinstance(cell, float):
        with pytest.raises(ValueError):
            result_cell, result_arg2 = function(cell, arg2=arg2)
    else:
        result_cell, result_arg2 = function(cell, arg2=arg2)

        # arg2 argument should be unchanged
        assert result_arg2 == arg2

        # check unit cell coercion to gemmi instance
        if cell is None:
            assert result_cell is None
        elif isinstance(cell, gemmi.UnitCell):
            assert np.array_equal(result_cell.parameters, cell.parameters)
        else:
            assert isinstance(result_cell, gemmi.UnitCell)
            expected = gemmi.UnitCell(*cell)
            assert np.array_equal(result_cell.parameters, expected.parameters)


@pytest.mark.parametrize("cell", [1, 2, 3])
@pytest.mark.parametrize(
    "arg2",
    [
        None,
        (1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
        [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
        np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
        gemmi.UnitCell(1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
        1.0,
    ],
)
def test_cellify_arg(cell, arg2):
    """Test cellify decorator with provided arguments"""

    if isinstance(arg2, float):
        with pytest.raises(ValueError):
            result_cell, result_arg2 = _function2_arg(cell, arg2=arg2)
    else:
        result_cell, result_arg2 = _function2_arg(cell, arg2=arg2)

        # cell argument should be unchanged
        assert result_cell == cell

        # check arg2 coercion to gemmi instance
        if arg2 is None:
            assert result_arg2 is None
        elif isinstance(arg2, gemmi.UnitCell):
            assert np.array_equal(result_arg2.parameters, arg2.parameters)
        else:
            assert isinstance(result_arg2, gemmi.UnitCell)
            expected = gemmi.UnitCell(*arg2)
            assert np.array_equal(result_arg2.parameters, expected.parameters)
