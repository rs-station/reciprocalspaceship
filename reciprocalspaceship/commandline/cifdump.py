#!/usr/bin/env python
"""
Summarize the contents of a CIF file.

Examples
--------
In order to summarize contents of file.cif::

    > rs.cifdump file.cif

If you would like to interactively inspect file.cif in an IPython
shell, use the "--embed" argument::

    > rs.cifdump file.cif --embed

If multiple CIF files are listed, they will be summarized sequentially,
and can be accessed in an IPython shell as a dictionary called `cifs`::

    > rs.cifdump file1.cif file2.cif file3.cif --embed

Usage Details
-------------
"""
import argparse

import pandas as pd

import reciprocalspaceship as rs

# If matplotlib is available, use pylab to setup IPython environment
try:
    from pylab import *
except:
    pass


def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument("cif", nargs="+", help="CIF file(s) to summarize")

    # Optional arguments
    parser.add_argument(
        "--embed",
        action="store_true",
        help=(
            "CIF file(s) will be summarized, and an IPython " "shell will be started"
        ),
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=3,
        help="Number of significant digits to output for floats",
    )

    return parser


def summarize(cif, precision):
    """Summarize contents of CIF file"""
    with pd.option_context("display.precision", precision):
        print(f"Spacegroup: {cif.spacegroup.short_name()}")
        print(f"Extended Hermann-Mauguin name: {cif.spacegroup.xhm()}")
        print(
            (
                f"Unit cell dimensions: {cif.cell.a:.3f} {cif.cell.b:.3f} {cif.cell.c:.3f} "
                f"{cif.cell.alpha:.3f} {cif.cell.beta:.3f} {cif.cell.gamma:.3f}"
            )
        )
        print(f"\ncif.head():\n\n{cif.head()}")
        print(f"\ncif.describe():\n\n{cif.describe()}")
        print(f"\ncif.dtypes:\n\n{cif.dtypes}")
    return


def main():
    # Parse commandline arguments
    parser = parse_arguments()
    args = parser.parse_args()

    if len(args.cif) == 1:
        cif = rs.read_cif(args.cif[0])
        summarize(cif, args.precision)
    else:
        cifs = dict(zip(args.cif, map(rs.read_cif, args.cif)))
        for key, value in cifs.items():
            print(f"CIF file: {key}\n")
            summarize(value, args.precision)
            print(f"{'-'*50}")

    # Begin IPython shell
    if args.embed:
        from IPython import embed

        bold = "\033[1m"
        end = "\033[0m"
        if "cifs" in locals():
            header = f"rs.DataSets stored in {bold}cifs{end} dictionary"
        else:
            header = f"rs.DataSet stored as {bold}cif{end}"
        print()
        embed(colors="neutral", header=header)

    return


if __name__ == "__main__":
    parser = main()
