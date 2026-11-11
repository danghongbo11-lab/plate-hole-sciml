import os, sys, csv, subprocess
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
ROOT=os.path.dirname(os.path.abspath(__file__)); NX=64; NY=32; L=4.; H=2.; CX=2.; CY=1.
rs=np.linspace(.20,.35,51); X,Y=np.meshgrid(np.linspace(0,L,NX),np.linspace(0,H,NY)); U1=[]; U2=[]; MASK=[]; rows=[]; failed=[]
for i,R in enumerate(rs):
    case='case_%03d'%i
    env=os.environ.copy(); env['PYFEM_R']=str(float(R))
    p=subprocess.run([sys.executable,os.path.join(ROOT,'solve_plate_hole.py')],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if p.returncode!=0:
        failed.append((case,R,p.stdout[-500:])); break
    d=pd.read_csv(os.path.join(ROOT,'nodal_displacement.csv')); pts=d[['x','y']].values
    m=((X-CX)**2+(Y-CY)**2>=R**2).astype(np.float64)
    z1=griddata(pts,d.U1.values,(X,Y),method='linear',fill_value=0.); z2=griddata(pts,d.U2.values,(X,Y),method='linear',fill_value=0.)
    z1[m==0]=0.; z2[m==0]=0.
    if not np.all(np.isfinite(z1)) or not np.all(np.isfinite(z2)): failed.append((case,R,'nonfinite')); break
    U1.append(z1); U2.append(z2); MASK.append(m); rows.append([case,R,float(np.max(z1)),float(np.max(np.abs(z2))),'SUCCESS'])
summary=os.path.join(ROOT,'dataset_summary.csv')
with open(summary,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['case_id','R','max_U1','max_abs_U2','status']); w.writerows(rows)
if failed: print('LAST_ERROR =',failed[0]); raise RuntimeError('dataset case failed')
np.savez(os.path.join(ROOT,'fno_dataset.npz'),R=rs,X=X,Y=Y,mask=np.array(MASK),U1=np.array(U1),U2=np.array(U2))
print('DATASET_STATUS = SUCCESS'); print('TOTAL_CASES =',len(rs)); print('SUCCESSFUL_CASES =',len(rows)); print('FAILED_CASES = 0'); print('R_MIN =',rs.min()); print('R_MAX =',rs.max()); print('DATASET_SHAPES =',np.array(U1).shape); print('R025_MAX_U1 =',rows[17][2]); print('NPZ_PATH =',os.path.join(ROOT,'fno_dataset.npz')); print('SUMMARY_CSV_PATH =',summary)
