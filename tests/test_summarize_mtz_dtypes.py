import pytest
import pandas as pd
import reciprocalspaceship as rs

@pytest.mark.parametrize("print_summary", [True, False])
def test_summarize_mtz_dtypes(capsys, print_summary):
    """Test rs.summarize_mtz_dtypes()"""
    df = rs.summarize_mtz_dtypes(print_summary)

    if print_summary:
        out, err = capsys.readouterr()
        out = out.splitlines()

        # stderr should be empty
        assert err == ""

        # check stdout
        assert len(out[0].split()) == 5
        assert len(out) == 18
        assert df is None
        for line in out[1:]:
            assert len(line.split()) == 4
    else:
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 17
