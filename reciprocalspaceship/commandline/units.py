#!/usr/bin/env python
"""
Useful unit conversions for crystallography

Examples
--------
To convert electron volts to Ångstroms,
    > rs.ev2angstroms 6_000

To convert Ångstroms to electron volts,
    > rs.angstroms2ev 1.0

"""
import argparse
import reciprocalspaceship as rs

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-p", "--precision", 
        help="The number of significant figures in the output with default 6.", 
        type=int, default=6,
    )
    return parser

def angstroms2ev():
    parser = get_parser()
    parser.add_argument(
        "wavelength", nargs="+", help="Photon wavelength in Ångstroms", type=float
    )
    parser = parser.parse_args()
    for w in parser.wavelength:
        print(rs.utils.angstroms2ev(w))

def ev2angstroms():
    parser = get_parser()
    parser.add_argument(
        "energy", nargs="+", help="Photon energy in electron volts", type=float)
    parser = parser.parse_args()
    for e in parser.energy:
        print(rs.utils.ev2angstroms(e))


