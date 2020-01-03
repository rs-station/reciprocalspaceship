import subprocess
import reciprocalspaceship as rs
import os

def localscale(crystal1, crystal2, sf_key1, err_key1, sf_key2, err_key2,
               inplace=False, cleanup=True):
    """
    Local scale crystal2 to crystal1 using SOLVE

    Parameters
    ----------
    crystal1 : Crystal
        'Native' crystal for local scaling
    crystal2 : Crystal
        'Derivative' crystal for local scaling
    sf_key1 : str
        Column name for structure factors in crystal1
    err_key1 : str
        Column name for error in structure factors in crystal1
    sf_key2 : str
        Column name for structure factors in crystal2
    err_key2 : str
        Column name for error in structure factors in crystal2
    inplace : bool
        Whether to overwrite crystal2 data after local scaling. If
        False, scaled Crystal object is returned, and crystal2 is 
        unchanged
    cleanup : bool
        Whether to delete intermediate files used by SOLVE

    Returns
    -------
    Crystal
         Local scaled Crystal object
    """

    # Assume crystal1 and crystal2 should have same spacegroup
    if crystal1.spacegroup.number != crystal2.spacegroup.number:
        raise ValueError(f"Spacegroup of Crystal objects must match for local scaling")

    # Write HKL files
    crystal1.write_hkl("native.hkl", sf_key1, err_key1)
    crystal2.write_hkl("derivative.hkl", sf_key2, err_key2)    
    files = ["native.hkl", "derivative.hkl"]
    
    # Write symmetry file
    symfile = f"{crystal1.spacegroup.short_name().lower()}.sym"
    with open(symfile, "w") as outsym:
        ops = crystal1.spacegroup.operations()
        outsym.write(f"{len(ops)}\n")
        for op in ops:
            outsym.write(f"{op.triplet()}\n")
    files.append(symfile)
            
    # Get high resolution limit
    high_res = 1.4              # TODO: Fix this
    
    # Write cryst file
    crystfile = "cryst.setup"
    files.append("cryst.setup")
    with open(crystfile, "w") as outcryst:
        cell = crystal1.cell
        outcryst.write(f"CELL {cell.a} {cell.b} {cell.c} {cell.alpha} {cell.beta} {cell.gamma}\n")
        outcryst.write(f"symfile {symfile}\n")
        outcryst.write(f"resolution {high_res} 200.0\n")

    # Convert HKL files to DORGBN files
    hkl2drg("native.hkl", "native.drg", crystfile)
    hkl2drg("derivative.hkl", "derivative.drg", crystfile)
    files.append("native.drg")
    files.append("derivative.drg")
    
    # Local scale derivative dataset to native dataset
    solve_localscale("native.drg", "derivative.drg", "scaled_derivative.drg", crystfile)
    files.append("scaled_derivative.drg")
    
    # Convert DORGBN file to HKL file
    drg2hkl("scaled_derivative.drg", "scaled_derivative.hkl", crystfile)
    files.append("scaled_derivative.hkl")
    
    # Read scaled reflections
    scaled = rs.read_hkl("scaled_derivative.hkl")

    hkls = scaled.index    
    if inplace:
        crystal2.drop(crystal2.index.difference(hkls), inplace=True)
        crystal2[sf_key2] = scaled["F"]
        crystal2[err_key2] = scaled["SigF"]
        F = crystal2
    else:
        F = crystal2.copy()
        F = F.loc[F.index.intersection(hkls)]
        F[sf_key2] = scaled["F"]
        F[err_key2] = scaled["SigF"]

    if cleanup:
        files.append("solve.ok")
        files.append("fem_io_unit_007")
        files.append("solve.prt")
        files.append("solve.logfile")
        files.append("solve.status")
        clean(files)
    
    return F

def clean(files):
    for f in files:
        os.remove(f)
    return

    
def hkl2drg(hklfile, drgfile, crystfile):
    """
    Convert HKL file to DORGBN file using SOLVE

    Parameters
    ----------
    hklfile : str
        Path to HKL file
    drgfile : str
        Path to which DORGBN file will be written
    crystfile : str
        Path to crystal setup file for SOLVE
    """
    
    cmd = f"""phenix.solve <<EOF
    @{crystfile}
    IMPORT
    {hklfile}
    1,        !  option 1 = one line per record
    {drgfile}
    TITLE
    2,        !  2 columns of data.
    Fobs
    Sigma
    1.0,      ! overall scale factor = 1.0
    y         ! Yes sort and map data
    n         ! No, do not swap indices
    0         ! don't interpret any columns as phases in degrees
    end
    EOF
    """

    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    return

def drg2hkl(drgfile, hklfile, crystfile):
    """
    Convert HKL file to DORGBN file using SOLVE

    Parameters
    ----------
    drgfile : str
        Path to DORGBN file
    hklfile : str
        Path to which HKL file will be written
    crystfile : str
        Path to crystal setup file for SOLVE
    """
    
    cmd = f"""phenix.solve <<EOF
    @{crystfile}
    INFILE  {drgfile}         !    input file name
    OUTFILE {hklfile}         !    output file name
    export
    end
    EOF
    """

    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    return

def solve_localscale(nativeDRG, derivativeDRG, scaledDRG, crystfile):

    cmd = f"""phenix.solve <<EOF
    @{crystfile}
    infile {nativeDRG}
    nnatf 1
    nnats 2
    infile(2) {derivativeDRG}
    nderf 1
    nders 2
    outfile {scaledDRG}
    
    tossbad          ! Lines DOEKE USED
    ratmin 0.5       ! Lines DOEKE USED
    ratio_out 10     ! Lines DOEKE USED

    localscale 
    end
    EOF
    """

    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    return
