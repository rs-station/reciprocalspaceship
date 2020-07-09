import pytest
import reciprocalspaceship as rs

def test_summarize_mtz_dtypes(capsys):
    """Test rs.summarize_mtz_dtypes()"""
    rs.print_mtz_dtypes()
    out, err = capsys.readouterr()
    out = out.splitlines()

    # stderr should be empty
    assert err == ""

    # check stdout
    assert len(out[0].split()) == 4
    assert len(out) == 18
    for line in out[1:]:
        assert len(line.split()) == 3
