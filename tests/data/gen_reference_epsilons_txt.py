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
dname = path.dirname(abspath) + "/epsilon_factors" 
if not path.exists(dname):                
    mkdir(dname)                          


outFN = dname + '/epsilon_factors.txt'

hmin,hmax = -5, 5
H = np.mgrid[hmin:hmax+1:1,hmin:hmax+1:1,hmin:hmax+1:1].reshape((3, -1)).T

with open(outFN, 'w') as f:
    f.write('xhm,hall,hm,h,k,l,epsilon\n')
    for s in tqdm(list(sgtbx.space_group_symbol_iterator())):
        hall = s.hall()
        hm   = s.hermann_mauguin()
        xhm  = s.universal_hermann_mauguin()
        sg   = sgtbx.space_group(s)
        for h in H:
            epsilon  = sg.epsilon(h)
            h,k,l = h
            f.write("{},{},{},{},{},{},{}\n".format(xhm,hall,hm,h,k,l,epsilon))

EOF

python - << EOF
import pandas as pd
from os import path,remove

abspath = path.abspath(__file__)          
dname = path.dirname(abspath) + "/epsilon_factors" 

inFN = dname + '/epsilon_factors.txt'

df = pd.read_csv(inFN)
df.to_csv(inFN + '.bz2', index=False)

remove(inFN)

EOF


