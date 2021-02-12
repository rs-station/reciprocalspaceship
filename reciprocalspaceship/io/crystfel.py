import pandas as pd
from reciprocalspaceship import DataSet
import numpy as np


def _parse_stream(filename: str) -> dict:
    """
    Parses stream and returns all indexed peak positions

    Parameters
    ----------
    filename : stream filename
        name of a .stream file 

    Returns
    --------
    (dict, np.ndarray)
    """

    answ_crystals = {}

    def contains_filename(s):
        return s.startswith("Image filename")

    def contains_event(s):
        return s.startswith("Event")

    def contains_serial_number(s):
        return s.startswith("Image serial number")

    def starts_chunk_peaks(s):
        return s.startswith("  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel")

    def ends_chunk_peaks(s):
        return s.startswith("End of peak list")

    def starts_crystal_peaks(s):
        return s.startswith(
            "   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel"
        )

    def is_photon_energy(s):
        return s.startswith('photon_energy_eV')

    def is_astar(s):
        return s.startswith('astar')

    def is_bstar(s):
        return s.startswith('bstar')

    def is_cstar(s):
        return s.startswith('cstar')

    def ends_crystal_peaks(s):
        return s.startswith("End of reflections")

    def eV2Angstrom(e_eV):
        return 12398. / e_eV

    # add unit cell parameters parsing
    with open(filename, 'r') as stream:
        is_unit_cell = False
        get_cellparam = lambda s: float(s.split()[2])
        rv_cell_param = None
        a, b, c, al, be, ga = [None]*6 # None's are needed since stream not always has all 6 parameters
        for line in stream:
            if 'Begin unit cell' in line:
                is_unit_cell = True
                continue
            elif is_unit_cell:
                if line.startswith('a ='):
                    a = get_cellparam(line)
                if line.startswith('b ='):
                    b = get_cellparam(line)
                if line.startswith('c ='):
                    c = get_cellparam(line)
                if line.startswith('al ='):
                    al = get_cellparam(line)
                if line.startswith('be ='):
                    be = get_cellparam(line)
                if line.startswith('ga ='):
                    ga = get_cellparam(line)
                    is_unit_cell = False # gamma is the last parameters
            elif 'End unit cell' in line:
                rv_cell_param = np.array([a, b, c, al, be, ga])
                break

    with open(filename, "r") as stream:
        is_chunk = False
        is_crystal = False
        current_filename = None
        current_event = None  # to handle non-event streams
        current_serial_number = None
        corrupted_chunk = False
        crystal_peak_number = 0
        crystal_idx = 0

        for line in stream:
            # analyzing what we have
            if ends_chunk_peaks(line):
                is_chunk = False
                chunk_peak_number = 0
            elif ends_crystal_peaks(line):
                is_crystal = False
                crystal_peak_number = 0

            elif is_photon_energy(line):
                photon_energy = float(line.split()[2])
            elif is_astar(line):
                astar = np.array(list(map(
                    float,
                    line.split()[2:5]))) / 10.  # crystfel's notation uses nm-1
            elif is_bstar(line):
                bstar = np.array(list(map(
                    float,
                    line.split()[2:5]))) / 10.  # crystfel's notation uses nm-1
            elif is_cstar(line):
                cstar = np.array(list(map(
                    float,
                    line.split()[2:5]))) / 10.  # crystfel's notation uses nm-1

                # since it's the last line needed to construct Ewald offset,
                # we'll pre-compute the matrices here
                A = np.array([astar, bstar, cstar]).T
                lambda_inv = 1 / eV2Angstrom(photon_energy)
                s0 = np.array([0, 0, lambda_inv]).T

            elif is_crystal:
                # example line:
                #    h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel
                #  -63   41    9     -41.31      57.45     195.00     170.86  731.0 1350.4 p0
                crystal_peak_number += 1
                h, k, l, I, sigmaI, _, _, _, _, _ = [i for i in line.split()]
                h, k, l = map(int, [h, k, l])

                # calculate ewald offset and s1
                hkl = np.array([h, k, l])
                s1 = A @ hkl + s0
                s1x, s1y, s1z = s1
                ewald_offset = np.linalg.norm(s1) - lambda_inv

                record = {
                    "H": h,
                    "K": k,
                    "L": l,
                    "I": float(I),
                    "sigmaI": float(sigmaI),
                    "BATCH": crystal_idx,
                    's1x': s1x,
                    's1y': s1y,
                    's1z': s1z,
                    'ewald_offset': ewald_offset
                }
                if current_event is not None:
                    name = (current_filename, current_event,
                            current_serial_number, crystal_idx,
                            crystal_peak_number)
                else:
                    name = (current_filename, current_serial_number,
                            crystal_idx, crystal_peak_number)
                answ_crystals[name] = record

            # start analyzing where we are now
            if corrupted_chunk:
                if "Begin chunk" not in line:
                    continue
                else:
                    is_crystal, is_chunk = False, False
                    corrupted_chunk = False
                    continue

            if contains_filename(line):
                current_filename = line.split()[-1]
            elif contains_event(line):
                current_event = line.split()[-1][2:]
            elif contains_serial_number(line):
                current_serial_number = line.split()[-1]

            elif starts_chunk_peaks(line):
                is_chunk = True
                continue

            elif starts_crystal_peaks(line):
                crystal_idx += 1
                is_crystal = True
                continue

    return answ_crystals, rv_cell_param


def read_crystfel(streamfile) -> DataSet:
    """
    Initialize attributes and populate the DataSet object with data from a CrystFEL stream with indexed reflections. This is the output format used by CrystFEL software when processing still diffraction data.

    Parameters
    ----------
    streamfile : stream filename
        name of a .stream file 
       
    Returns
    --------
    rs.DataSet
    """

    if not streamfile.endswith('.stream'):
        raise ValueError("Stream file should end with .stream")
    # read data from stream file
    d, cell = _parse_stream(streamfile)
    df = pd.DataFrame.from_records(list(d.values()))

    # set mtztypes as in precognition.py
    # hkl -- H
    # I, sigmaI -- J, Q
    # BATCH -- B
    # s1{x,y,z} -- R
    # ewald_offset -- R
    names = [
        'H', 'K', 'L', 'I', 'sigmaI', 'BATCH', 's1x', 's1y', 's1z',
        'ewald_offset'
    ]
    mtztypes = ["H", "H", "H", "J", "Q", "B", "R", "R", "R", "R"]
    dataset = DataSet()
    for (k, v), mtztype in zip(df.items(), mtztypes):
        dataset[k] = v.astype(mtztype)
    dataset.set_index(['H', 'K', 'L'], inplace=True)

    dataset.merged = False  # CrystFEL stream is always unmerged
    dataset.cell = cell

    return dataset
