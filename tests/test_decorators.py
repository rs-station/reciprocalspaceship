import gemmi
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


@pytest.mark.parametrize("sg", [None, 1, gemmi.SpaceGroup(1), "P 1", 1.0])
@pytest.mark.parametrize("extra_arg", [1, 2, 3])
@pytest.mark.parametrize("function", [_function_bare, _function_empty])
def test_spacegroupify_default(sg, function, extra_arg):
    """Test spacegroupify decorator with default arguments"""

    if isinstance(sg, float):
        with pytest.raises(ValueError):
            result_spacegroup, result_arg2 = function(sg, arg2=extra_arg)
    else:
        result_spacegroup, result_arg2 = function(sg, arg2=extra_arg)

        # arg2 argument should be unchanged
        assert result_arg2 == extra_arg

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
