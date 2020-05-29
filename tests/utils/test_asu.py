import pytest
import numpy as np
import reciprocalspaceship as rs
import gemmi

def test_in_asu(sgtbx_by_xhm):
    """
    Test rs.utils.in_asu using reference data generated from sgtbx
    """
    xhm = sgtbx_by_xhm[0]
    reference  = sgtbx_by_xhm[1]

    H = reference[['h', 'k', 'l']].to_numpy()
    sg = gemmi.SpaceGroup(xhm)
    in_asu = rs.utils.in_asu(H, sg)
    ref_in_asu = reference['in_asu'].to_numpy()
    assert np.array_equal(in_asu, ref_in_asu)


@pytest.mark.parametrize("return_phase_shifts", [True, False])
def test_hkl_to_asu(sgtbx_by_xhm, return_phase_shifts):
    """
    Test rs.utils.hkl_to_asu using reference data generated from sgtbx
    """
    xhm = sgtbx_by_xhm[0]
    reference  = sgtbx_by_xhm[1]

    # Test rs.utils.hkl_to_asu() reciprocalspace ASU against sgtbx
    H = reference[['h', 'k', 'l']].to_numpy()
    sg = gemmi.SpaceGroup(xhm)
    if return_phase_shifts:
        Hasu_rs, isym, phic, phis = rs.utils.hkl_to_asu(H, sg, return_phase_shifts)
    else:
        Hasu_rs, isym = rs.utils.hkl_to_asu(H, sg)
    Hasu_sgtbx = reference[['h_asu', 'k_asu', 'l_asu']].to_numpy()
    assert np.array_equal(Hasu_rs, Hasu_sgtbx)

    def build_emptyMTZ(sg):
        """Build empty gemmi.MTZ"""
        m = gemmi.Mtz()
        m.spacegroup = sg
        m.add_dataset('crystal')
        m.add_column('H', 'H')
        m.add_column('K', 'H')
        m.add_column('L', 'H')
        m.add_column('M/ISYM', 'Y')
        return m

    def gemmi_map2asu(H, sg, return_phase_shifts):
        """Map HKLs to reciprocal space ASU using gemmi"""
        m = build_emptyMTZ(sg)
        data = np.append(H, np.zeros((len(H), 1)), 1)
        m.set_data(data)
        m.switch_to_asu_hkl()
        h = m.column_with_label('H').array.astype(int)
        k = m.column_with_label('K').array.astype(int)
        l = m.column_with_label('L').array.astype(int)
        ref_isym = m.column_with_label('M/ISYM').array.astype(int)

        if return_phase_shifts:
            ops_plus = sg.operations().sym_ops
            ops_minus = [op.negated() for op in sg.operations().sym_ops]
            ops = np.array(list(zip(ops_plus, ops_minus))).flatten()
            shift = [ops[i-1].phase_shift(h) for i, h in zip(ref_isym, H)]
            shift = rs.utils.canonicalize_phases(np.array(shift), deg=False)
            return np.vstack([h, k, l]).T, ref_isym, shift
        
        return np.vstack([h, k, l]).T, ref_isym

    # Test M/ISYM and phase shift against gemmi for cases where gemmi
    # matches sgtbx
    if return_phase_shifts:
        Hasu_gemmi, ref_isym, ref_phis = gemmi_map2asu(H, sg, return_phase_shifts)
        phis = np.deg2rad(rs.utils.canonicalize_phases(phic*phis))
    else:
        Hasu_gemmi, ref_isym = gemmi_map2asu(H, sg, return_phase_shifts)

    assert np.array_equal(isym, ref_isym)
    if return_phase_shifts:
        assert np.isclose(np.sin(phis), np.sin(ref_phis)).all()
        assert np.isclose(np.cos(phis), np.cos(ref_phis)).all()

