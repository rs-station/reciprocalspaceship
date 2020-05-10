#!/bin/bash

cctbx.python - << EOF
from cctbx import sgtbx
from cctbx import miller
import numpy as np
from os import path,mkdir

try:                                      
    from tqdm import tqdm                 
except:                                   
    tqdm = iter                           
                                          
                                          
#Run in the fmodel directory              
abspath = path.abspath(__file__)          
dname = path.dirname(abspath) + "/asu" 
if not path.exists(dname):                
    mkdir(dname)                          

outFN = dname + '/inside.csv'


hmin,hmax = -5, 5
H = np.mgrid[hmin:hmax+1:1,hmin:hmax+1:1,hmin:hmax+1:1].reshape((3, -1)).T


with open(outFN, 'w') as f:
    f.write("hall, h, k, l, in_asu\n")
    for s in tqdm(list(sgtbx.space_group_symbol_iterator())):
        hall = s.hall()
        sg   = sgtbx.space_group(hall)
        asu = sgtbx.reciprocal_space_asu(sg.type())
        in_asu = [asu.is_inside(i) for i in H]
        for h,value in zip(H, in_asu):
            f.write(','.join([
                hall,
                str(h[0]),
                str(h[1]),
                str(h[2]),
                str(value),
            ]) + '\n')

outFN = dname + '/remapped.csv'

with open(outFN, 'w') as f:
    f.write("hall, h, k, l, h_asu, k_asu, l_asu\n")
    for s in tqdm(list(sgtbx.space_group_symbol_iterator())):
        hall = s.hall()
        sg   = sgtbx.space_group(hall)
        asu = sgtbx.reciprocal_space_asu(sg.type())
        H_asu = np.vstack([miller.asym_index(sg, asu, i).h() for i in H])
        for h,h_asu in zip(H, H_asu):
            f.write(','.join([
                hall,
                str(h[0]),
                str(h[1]),
                str(h[2]),
                str(h_asu[0]),
                str(h_asu[1]),
                str(h_asu[2]),
            ]) + '\n')

EOF

python - << EOF

import pandas as pd
from os import remove,path

#Run in the fmodel directory              
abspath = path.abspath(__file__)          
dname = path.dirname(abspath) + "/asu" 

inFN = dname + '/inside.csv'

df = pd.read_csv(inFN)
df.to_csv(inFN + '.bz2', index=False)

remove(inFN)

inFN = dname + '/remapped.csv'

df = pd.read_csv(inFN)
df.to_csv(inFN + '.bz2', index=False)

remove(inFN)

EOF
