import os
from io import StringIO
from os import devnull, mkdir, path, rename
from subprocess import call

import pandas as pd
from urllib3 import PoolManager

try:
    from tqdm import tqdm
except:
    tqdm = iter


# Run in the fmodel directory
abspath = path.abspath(__file__)
dname = path.dirname(abspath) + "/fmodel"
if not path.exists(dname):
    mkdir(dname)
os.chdir(dname)


high_resolution = 8.0


# KMD has removed 2BOP and 1D00 from the data directory
# they are weird space groups and not passing tests.
# this could be an issue with the pdb files or with the
# the software -- should debug this later.
# Add these two lines back in for teting:
# 2BOP, 155, R 3 2
# 1D00, 146, R 3
pdbs = pd.read_csv(
    StringIO(
        """PDBID, space group number, space group name
6gl4, 3, P 1 2 1
6ofl, 4, P 1 21 1
6h7c, 5, C 1 2 1
2z51, 16, P 2 2 2
6a8k, 17, P 2 2 21
6nsv, 18, P 2 21 21
3kxe, 19, P 21 21 21
1oel, 20, C 2 2 21
6E1X, 21, C 2 2 2
6fxw, 22, F 2 2 2
3ruw, 23, I 2 2 2
5CR4, 24, I 21 21 21
2ON8, 75, P 4
6E6T, 76, P 41
4AK8, 77, P 42
6E6N, 78, P 43
3t4d, 79, I 4
6GUS, 80, I 41
5ZOA, 89, P 4 2 2
5Z3A, 90, P 4 21 2
5QRH, 91, P 41 2 2
6H9J, 92, P 41 21 2
6Q8D, 93, P 42 2 2
6O11, 94, P 42 21 2
6OH9, 95, P 43 2 2
9lyz, 96, P 43 21 2
6NMT, 97, I 4 2 2
6NRH, 98, I 41 2 2
6mbu, 143, P 3
6ITG, 144, P 31
6JD9, 145, P 32
6NEN, 149, P 3 1 2
6M9W, 150, P 3 2 1
6NPT, 151, P 31 1 2
6b8z, 152, P 31 2 1
5w79, 154, P 32 2 1
6e02, 168, P 6
6ovt, 169, P 61
6PDI, 170, P 65
6H64, 171, P 62
6DWF, 172, P 64
6GJ6, 173, P 63
6CY6, 177, P 6 2 2
6Q58, 178, P 61 2 2
6GEO, 179, P 65 2 2
6Q1Y, 180, P 62 2 2
6I25, 181, P 64 2 2
6IWV, 182, P 63 2 2
6D9T, 195, P 2 3
6QLH, 196, F 2 3
5YUP, 197, I 2 3
6ITP, 198, P 21 3
6S34, 199, I 21 3
6ECB, 207, P 4 3 2
6R62, 208, P 42 3 2
6I9P, 209, F 4 3 2
4cy9, 210, F 41 3 2
6D3B, 211, I 4 3 2
6EDM, 212, P 43 3 2
6CN8, 213, P 41 3 2
4I6Y, 214, I 41 3 2
"""
    ),
    dtype={"PDBID": str, "space group number": int, "space group name": str},
)


for pdbid in tqdm(pdbs["PDBID"]):
    pdbid = pdbid.upper()
    with open(devnull, "w") as null:
        call(
            f"wget files.rcsb.org/download/{pdbid}.pdb".split(),
            stdout=null,
            stderr=null,
        )
        pdbFN = f"{pdbid}.pdb"
        url = f"files.rcsb.org/download/{pdbFN}"
        with PoolManager() as http:
            r = http.request("GET", url)
            with open(pdbFN, "wb") as out:
                out.write(r.data)
        call(
            f"phenix.fmodel {pdbFN} high_resolution={high_resolution}".split(),
            stdout=null,
        )
        rename(f"{pdbFN}.mtz", f"{pdbid}.mtz")
        call(
            f"phenix.reflection_file_converter {pdbid}.mtz --expand_to_p1 --mtz={pdbid}_p1.mtz".split(),
            stdout=null,
        )


# Run in the r3 directory (this space group is problematic for gemmi)
dname = path.dirname(abspath) + "/r3"
if not path.exists(dname):
    mkdir(dname)
os.chdir(dname)

pdbs = pd.read_csv(
    StringIO(
        """PDBID, space group number, space group name
1CTJ, 146, R 3
2QXX, 155, R 3 2
"""
    ),
    dtype={"PDBID": str, "space group number": int, "space group name": str},
)


for pdbid in tqdm(pdbs["PDBID"]):
    pdbid = pdbid.upper()
    with open(devnull, "w") as null:
        call(
            f"wget files.rcsb.org/download/{pdbid}.pdb".split(),
            stdout=null,
            stderr=null,
        )
        pdbFN = f"{pdbid}.pdb"
        url = f"files.rcsb.org/download/{pdbFN}"
        with PoolManager() as http:
            r = http.request("GET", url)
            with open(pdbFN, "wb") as out:
                out.write(r.data)
        call(
            f"phenix.fmodel {pdbFN} high_resolution={high_resolution}".split(),
            stdout=null,
        )
        rename(f"{pdbFN}.mtz", f"{pdbid}.mtz")
        call(
            f"phenix.reflection_file_converter {pdbid}.mtz --expand_to_p1 --mtz={pdbid}_p1.mtz".split(),
            stdout=null,
        )
