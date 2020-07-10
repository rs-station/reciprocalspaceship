#!/usr/bin/env python
"""
Summarize the contents of an MTZ file. 

Examples
--------
In order to summarize contents of file.mtz::

    > rs.mtzdump file.mtz

If you would like to interactively inspect file.mtz in an IPython 
shell, use the "--embed" argument::

    > rs.mtzdump file.mtz --embed

Usage Details
-------------
"""
import argparse
import pandas as pd
import reciprocalspaceship as rs

def parse_arguments():
    """Parse commandline arguments"""
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=__doc__)

    # Required arguments
    parser.add_argument("mtz", help="MTZ file to summarize")

    # Optional arguments
    parser.add_argument("--embed", action='store_true',
                        help=("MTZ file will be summarized, and an IPython "
                              "shell will be started"))
    parser.add_argument("-p", "--precision", type=int, default=3,
                        help="Number of significant digits to output for floats")
    
    return parser

def main():

    # Parse commandline arguments
    parser = parse_arguments()
    args = parser.parse_args()

    # Summarize contents of MTZ
    mtz = rs.read_mtz(args.mtz)
    with pd.option_context('display.precision', args.precision):
        print(f"Spacegroup: {mtz.spacegroup.short_name()}")
        print(f"Extended Hermann-Mauguin name: {mtz.spacegroup.xhm()}")
        print((f"Unit cell dimensions: {mtz.cell.a:.3f} {mtz.cell.b:.3f} {mtz.cell.c:.3f} "
               f"{mtz.cell.alpha:.3f} {mtz.cell.beta:.3f} {mtz.cell.gamma:.3f}"))
        print(f"\nmtz.head():\n\n{mtz.head()}")
        print(f"\nmtz.describe():\n\n{mtz.describe()}")
        print(f"\nmtz.dtypes:\n\n{mtz.dtypes}")    
        
    # Begin IPython shell
    if args.embed:
        from IPython import embed
        bold = '\033[1m'
        end  = '\033[0m'
        header = f"rs.DataSet stored as {bold}mtz{end}"
        print()
        embed(colors='neutral', header=header)

    return
    
if __name__ == "__main__":
    parser = main()
