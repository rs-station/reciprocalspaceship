import pytest
import gemmi

@pytest.fixture(params=[gemmi.SpaceGroup(n) for n in [1, 4, 5, 19, 152]])
def common_spacegroup(request):
    """Yields common space groups for macromolecular crystals"""
    return request.param
