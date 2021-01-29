#!/usr/bin/env cctbx.python

from cctbx.french_wilson import fw_centric,fw_acentric
import pandas as pd
import numpy as np

outFN = "french_wilson/fw_test_data.csv"

Imin,Imax,Istep = -3, 10, 1.0
SigImin,SigImax,SigIstep = 0.1, 10, 1.0
Sigmamin,Sigmamax,Sigmastep = 0.1, 10., 1.0

I,SigI,Sigma = np.mgrid[Imin:Imax:Istep,SigImin:SigImax:SigIstep,Sigmamin:Sigmamax:Sigmastep].reshape((3, -1))

data = []
for args in zip(I, SigI, Sigma):
    i,sigi,sigma = args
    result = fw_acentric(i, sigi, sigma, -4.)
    if result != (-1., -1., -1., -1.): #These were rejected
        data.append(args + result)

data = np.vstack(data)
df = pd.DataFrame(data = data, columns=('I', 'SigI', 'Sigma', 'FW-I', 'FW-SigI', 'FW-F', 'FW-SigF'))
df['CENTRIC'] = False

data = []
for args in zip(I, SigI, Sigma):
    i,sigi,sigma = args
    result = fw_centric(i, sigi, sigma, -4.)
    if result != (-1., -1., -1., -1.): #These were rejected
        data.append(args + result)

data = np.vstack(data)
_df = pd.DataFrame(data = data, columns=('I', 'SigI', 'Sigma', 'FW-I', 'FW-SigI', 'FW-F', 'FW-SigF'))
_df['CENTRIC'] = True

df = df.append(_df)
df.to_csv(outFN)

