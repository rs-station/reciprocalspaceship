import reciprocalspaceship as rs
from os.path import abspath


inFN = '/'.join(abspath(__file__).split('/')[:-1]) + \
    "/data/fmodel/3KXE.mtz"


dataset = rs.read_mtz(inFN)


from IPython import embed
embed(colors='neutral')
