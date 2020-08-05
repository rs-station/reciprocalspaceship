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

If multiple MTZ files are listed, they will be summarized sequentially,
and can be accessed in an IPython shell as a dictionary called `mtzs`::

    > rs.mtzdump file1.mtz file2.mtz file3.mtz --embed

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
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=__doc__)

    # Required arguments
    parser.add_argument("mtz", nargs='+', help="MTZ file(s) to summarize")

    # Optional arguments
    parser.add_argument("--embed", action='store_true',
                        help=("MTZ file(s) will be summarized, and an IPython "
                              "shell will be started"))
    parser.add_argument("-p", "--precision", type=int, default=3,
                        help="Number of significant digits to output for floats")
    
    return parser

def summarize(mtz, precision):
    """Summarize contents of MTZ file"""
    with pd.option_context('display.precision', precision):
        print(f"Spacegroup: {mtz.spacegroup.short_name()}")
        print(f"Extended Hermann-Mauguin name: {mtz.spacegroup.xhm()}")
        print((f"Unit cell dimensions: {mtz.cell.a:.3f} {mtz.cell.b:.3f} {mtz.cell.c:.3f} "
               f"{mtz.cell.alpha:.3f} {mtz.cell.beta:.3f} {mtz.cell.gamma:.3f}"))
        print(f"\nmtz.head():\n\n{mtz.head()}")
        print(f"\nmtz.describe():\n\n{mtz.describe()}")
        print(f"\nmtz.dtypes:\n\n{mtz.dtypes}")    
    return

def main():

    # Parse commandline arguments
    parser = parse_arguments()
    args = parser.parse_args()

    if len(args.mtz) == 1:
        mtz = rs.read_mtz(args.mtz[0])
        summarize(mtz, args.precision)
    else:
        mtzs =  dict(zip(args.mtz, map(rs.read_mtz, args.mtz)))
        for key, value in mtzs.items():
            print(f"MTZ file: {key}\n")
            summarize(value, args.precision)
            print(f"{'-'*50}")
            
    # Begin IPython shell
    if args.embed:
        from IPython import embed
        bold = '\033[1m'
        end  = '\033[0m'
        if "mtzs" in locals():
            header = f"rs.DataSets stored in {bold}mtzs{end} dictionary"
        else:
            header = f"rs.DataSet stored as {bold}mtz{end}"
        print()
        embed(colors='neutral', header=header)

    return
    
if __name__ == "__main__":
    parser = main()
