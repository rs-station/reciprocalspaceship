#!/usr/bin/env cctbx.python

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
dname = path.dirname(abspath) + "/systematic_absences" 
if not path.exists(dname):                
    mkdir(dname)                          


outFN = dname + '/systematic_absences.txt'


hmin,hmax = -3, 3
h = np.arange(hmin, hmax+1)
H = np.array(np.meshgrid(h, h, h))
H = H.reshape(3, (hmax - hmin + 1)**3).T


with open(outFN, 'w') as f:
    f.write('h' + ',' + ','.join(map(str, H[:,0])) + '\n')
    f.write('k' + ',' + ','.join(map(str, H[:,1])) + '\n')
    f.write('l' + ',' + ','.join(map(str, H[:,2])) + '\n')
    for s in tqdm(list(sgtbx.space_group_symbol_iterator())):
        hall = s.hall()
        sg   = sgtbx.space_group(hall)
        absent  = [sg.is_sys_absent(i) for i in H]
        f.write(hall + ',' + ','.join(map(str, absent)) + '\n')
