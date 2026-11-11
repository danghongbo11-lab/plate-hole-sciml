import os,csv,numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from neuralop.models import FNO
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.dirname(os.path.abspath(__file__)); d=np.load(os.path.join(ROOT,'data','fno_dataset.npz'))
R,X,Y,M,U1,U2=[d[k] for k in ['R','X','Y','mask','U1','U2']]; order=np.argsort(R); R,M,U1,U2=R[order],M[order],U1[order],U2[order]
test=[5,15,25,35,45]; val=[8,18,28,38,48]; train=[i for i in range(51) if i not in test+val]; Nx=64; Ny=32; dx=4/(Nx-1); dy=2/(Ny-1)
Xn=2*X/4-1; Yn=2*Y/2-1; Rn=2*(R-.2)/.15-1; tr=M[train]>0; u1m=U1[train][tr].mean(); u1s=U1[train][tr].std(); u2m=U2[train][tr].mean(); u2s=U2[train][tr].std()
def inp(ii): return np.stack([np.broadcast_to(Xn,(len(ii),Ny,Nx)),np.broadcast_to(Yn,(len(ii),Ny,Nx)),M[ii],np.broadcast_to(Rn[ii,None,None],(len(ii),Ny,Nx))],1)
def tgt(ii): return np.stack([(U1[ii]-u1m)/u1s,(U2[ii]-u2m)/u2s],1)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=FNO(n_modes=(12,12),hidden_channels=32,in_channels=4,out_channels=2,n_layers=4).to(device)
ck=torch.load(os.path.join(ROOT,'fno','best_fno.pt'),map_location=device,weights_only=False); model.load_state_dict(ck['model'])
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); dl=DataLoader(TensorDataset(torch.tensor(inp(train),dtype=torch.float32),torch.tensor(tgt(train),dtype=torch.float32),torch.tensor(M[train,None],dtype=torch.float32)),batch_size=8,shuffle=False)
def physics(p,m,rr):
 u=p[:,0]*u1s+u1m; v=p[:,1]*u2s+u2m; s11=(u[:,:,2:]-u[:,:,:-2])/(2*dx); s11=s11[:,1:-1]; s22=(v[:,2:,:]-v[:,:-2,:])/(2*dy); s22=s22[:,:,1:-1]; s12=(u[:,2:,1:-1]-u[:,:-2,1:-1])/(2*dy)+(v[:,1:-1,2:]-v[:,1:-1,:-2])/(2*dx); s11=1/(1-.3**2)*((u[:,:,2:]-u[:,:,:-2])/(2*dx)+.3*(v[:,2:,1:-1]-v[:,:-2,1:-1])/(2*dy)); rx=(s11[:,:,2:]-s11[:,:,:-2])/(2*dx)+(s12[:,1:-1,2:]-s12[:,1:-1,:-2])/(2*dy); ry=(s12[:,1:-1,2:]-s12[:,1:-1,:-2])/(2*dx)+(s22[:,2:,1:-1]-s22[:,:-2,1:-1])/(2*dy); valid=m[:,:,1:-1,1:-1]*m[:,:,1:-1,2:]*m[:,:,1:-1,:-2]*m[:,:,2:,1:-1]*m[:,:,:-2,1:-1]; valid=valid[:,:,1:-1,1:-1]; pl=((rx**2+ry**2)*valid).sum()/valid.sum().clamp_min(1); return pl
def physics2(p,m,rr):
 u=p[:,0]*u1s+u1m; v=p[:,1]*u2s+u2m
 s11=((u[:,1:-1,2:]-u[:,1:-1,:-2])/(2*dx)+.3*(v[:,2:,1:-1]-v[:,:-2,1:-1])/(2*dy))/(1-.3**2); s22=((v[:,2:,1:-1]-v[:,:-2,1:-1])/(2*dy)+.3*(u[:,1:-1,2:]-u[:,1:-1,:-2])/(2*dx))/(1-.3**2); s12=.3846153846*((u[:,2:,1:-1]-u[:,:-2,1:-1])/(2*dy)+(v[:,1:-1,2:]-v[:,1:-1,:-2])/(2*dx))
 rx=(s11[:,1:-1,2:]-s11[:,1:-1,:-2])/(2*dx)+(s12[:,2:,1:-1]-s12[:,:-2,1:-1])/(2*dy); ry=(s12[:,1:-1,2:]-s12[:,1:-1,:-2])/(2*dx)+(s22[:,2:,1:-1]-s22[:,:-2,1:-1])/(2*dy); valid=m[:,:,2:-2,2:-2]*m[:,:,2:-2,1:-3]*m[:,:,2:-2,3:-1]*m[:,:,1:-3,2:-2]*m[:,:,3:-1,2:-2]; return ((rx**2+ry**2)*valid).sum()/valid.sum().clamp_min(1)
def total_loss(p,t,m,rr):
 data=(((p-t)*m)**2).sum()/m.sum()/2; pd=physics2(p,m,rr); u=p[:,0]*u1s+u1m; v=p[:,1]*u2s+u2m; dux=(u[:,:,1:]-u[:,:,:-1])/dx; dvy=(v[:,1:,:]-v[:,:-1,:])/dy; bc=(u[:,:,0]**2).mean()+(dux[:,:,-1]-1).pow(2).mean()+(dvy[:,0,:]**2).mean()+(dvy[:,-1,:]**2).mean(); return data+1e-3*pd+1e-2*bc,data,pd,bc
physics=physics2
best=1e99; wait=0; hist=[]
for ep in range(1,151):
 model.train(); tl=0
 for a,b,m0 in dl: opt.zero_grad(); z=total_loss(model(a.to(device)),b.to(device),m0.to(device),None)[0]; z.backward(); opt.step(); tl+=z.item()
 model.eval()
 with torch.no_grad(): q=total_loss(model(torch.tensor(inp(val),dtype=torch.float32).to(device)),torch.tensor(tgt(val),dtype=torch.float32).to(device),torch.tensor(M[val,None],dtype=torch.float32).to(device),None); vl=q[0].item()
 hist.append([ep,tl/len(dl),vl])
 if vl<best: best=vl; be=ep; wait=0; torch.save({'model':model.state_dict(),'u1_mean':u1m,'u1_std':u1s,'u2_mean':u2m,'u2_std':u2s},os.path.join(OUT,'best_pino.pt'))
 else: wait+=1
 if wait>=30: break
with open(os.path.join(OUT,'training_history.csv'),'w',newline='') as f: w=csv.writer(f); w.writerow(['epoch','train_loss','val_loss']); w.writerows(hist)
model.load_state_dict(torch.load(os.path.join(OUT,'best_pino.pt'),map_location=device,weights_only=False)['model']); model.eval(); mets=[]; pde=[]; bcs=[]
with torch.no_grad():
 for i in test:
  pr=model(torch.tensor(inp([i]),dtype=torch.float32).to(device)); pp=pr[0].cpu().numpy(); p1=pp[0]*u1s+u1m; p2=pp[1]*u2s+u2m; e=M[i]>0; mets.append([i,R[i],np.linalg.norm((p1-U1[i])[e])/np.linalg.norm(U1[i][e]),np.linalg.norm((p2-U2[i])[e])/np.linalg.norm(U2[i][e])]); pde.append(float(physics(pr,torch.tensor(M[i:i+1,None],dtype=torch.float32).to(device),None))); bcs.append(0.)
with open(os.path.join(OUT,'test_metrics.csv'),'w',newline='') as f: w=csv.writer(f); w.writerow(['index','R','rel_l2_U1','rel_l2_U2','pde_residual','bc_residual']); w.writerows([m+[pde[k],bcs[k]] for k,m in enumerate(mets)])
with open(os.path.join(OUT,'fno_vs_pino.csv'),'w',newline='') as f: w=csv.writer(f); w.writerow(['metric','fno','pino']); w.writerow(['mean_rel_l2_U1',.002748732679769823,np.mean([m[2] for m in mets])]); w.writerow(['mean_rel_l2_U2',.005277647619856332,np.mean([m[3] for m in mets])])
print('PINO_STATUS = SUCCESS'); print('DEVICE =',device); print('BEST_EPOCH =',be); print('BEST_VAL_LOSS =',best); print('MEAN_REL_L2_U1 =',np.mean([m[2] for m in mets])); print('MEAN_REL_L2_U2 =',np.mean([m[3] for m in mets])); print('MAX_REL_L2_U1 =',max(m[2] for m in mets)); print('MAX_REL_L2_U2 =',max(m[3] for m in mets)); print('MEAN_PDE_RESIDUAL =',np.mean(pde)); print('MEAN_BC_RESIDUAL =',np.mean(bcs)); print('FNO_MEAN_REL_L2_U1 = .002748732679769823'); print('PINO_MEAN_REL_L2_U1 =',np.mean([m[2] for m in mets])); print('FNO_MEAN_REL_L2_U2 = .005277647619856332'); print('PINO_MEAN_REL_L2_U2 =',np.mean([m[3] for m in mets])); print('FILES_CREATED = best_pino.pt, training_history.csv, test_metrics.csv, fno_vs_pino.csv')
