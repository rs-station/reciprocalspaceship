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
dname = path.dirname(abspath) + "/sgtbx" 
if not path.exists(dname):                
    mkdir(dname)                          


hmin,hmax = -5, 5
H = np.mgrid[hmin:hmax+1:1,hmin:hmax+1:1,hmin:hmax+1:1].reshape((3, -1)).T


outFN = dname + '/sgtbx.csv'

with open(outFN, 'w') as f:
    f.write("xhm,hall,hm,h,k,l,h_asu,k_asu,l_asu,in_asu,is_centric,is_absent,epsilon\n")
    for s in tqdm(list(sgtbx.space_group_symbol_iterator())):
        hall = s.hall()
        hm   = s.hermann_mauguin()
        xhm  = s.universal_hermann_mauguin()
        sg   = sgtbx.space_group(s)
        asu = sgtbx.reciprocal_space_asu(sg.type())
        H_asu = np.vstack([miller.asym_index(sg, asu, i).h() for i in H])
        in_asu = [asu.is_inside(i) for i in H]
    	is_centric = [sg.is_centric(h) for h in H]
	is_absent  =  [sg.is_sys_absent(h) for h in H]
	epsilons   =  [sg.epsilon(h) for h in H]
        for h,h_asu,inside,centric,absent,epsilon in zip(H, H_asu, in_asu, is_centric, is_absent, epsilons):
            f.write(','.join([
                xhm,
                hall,
                hm,
                str(h[0]),
                str(h[1]),
                str(h[2]),
                str(h_asu[0]),
                str(h_asu[1]),
                str(h_asu[2]),
                str(inside),
		str(centric),
		str(absent),
		str(epsilon),
            ]) + '\n')

EOF

python - << EOF

import gemmi 
import numpy as np
import pandas as pd
from os import remove,path

#Run in the fmodel directory
abspath = path.abspath(__file__)
dname = path.join(path.dirname(abspath), "sgtbx")

inFN = path.join(dname, 'sgtbx.csv')

df = pd.read_csv(inFN)

df['isym'] = 0

def isym(df):
    xhm = df['xhm'].iloc[0]
    sg = gemmi.SpaceGroup(xhm)
    m = gemmi.Mtz()
    m.spacegroup = sg
    m.add_dataset('crystal')
    m.add_column('H', 'H')
    m.add_column('K', 'H')
    m.add_column('L', 'H')
    m.add_column('M/ISYM', 'Y')
    m.set_data(df[['h', 'k', 'l', 'isym']].to_numpy(dtype=np.float32))
    success = m.switch_to_asu_hkl()

    if not success:
        raise ValueError("Mtz.switch_to_asu_hkl retval was False! Failed to map to ASU")

    df['h_gemmi'] = m.column_with_label('H').array.astype(int)
    df['k_gemmi'] = m.column_with_label('K').array.astype(int)
    df['l_gemmi'] = m.column_with_label('L').array.astype(int)
    df['isym']    = m.column_with_label('M/ISYM').array.astype(int)
    return df

result = df.groupby('xhm', as_index=False).apply(isym)
result.to_csv(inFN + '.bz2', index=False)
remove(inFN)

EOF
