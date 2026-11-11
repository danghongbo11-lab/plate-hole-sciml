import os,csv,numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset
from neuralop.models import FNO
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.dirname(os.path.abspath(__file__)); d=np.load(os.path.join(ROOT,'data','fno_dataset.npz'))
R,X,Y,mask,U1,U2=[d[k] for k in ['R','X','Y','mask','U1','U2']]
assert R.shape==(51,) and X.shape==(32,64) and Y.shape==(32,64) and mask.shape==(51,32,64) and U1.shape==(51,32,64) and U2.shape==(51,32,64)
assert np.isnan(U1).sum()+np.isnan(U2).sum()+np.isinf(U1).sum()+np.isinf(U2).sum()==0
order=np.argsort(R); R,M,V1,V2=R[order],mask[order],U1[order],U2[order]; Xn=2*X/4-1; Yn=2*Y/2-1; Rn=2*(R-.20)/.15-1
test=[5,15,25,35,45]; val=[8,18,28,38,48]; train=[i for i in range(51) if i not in test+val]
print('MIN_ABS_R_MINUS_025 =',np.min(abs(R-.25))); print('CLOSEST_R_TO_025 =',R[np.argmin(abs(R-.25))]); print('TRAIN_R =',R[train]); print('VAL_R =',R[val]); print('TEST_R =',R[test])
trmask=M[train]>0; u1m=V1[train][trmask].mean(); u1s=V1[train][trmask].std(); u2m=V2[train][trmask].mean(); u2s=V2[train][trmask].std()
def inp(ii): return np.stack([np.broadcast_to(Xn,(len(ii),32,64)),np.broadcast_to(Yn,(len(ii),32,64)),M[ii],np.broadcast_to(Rn[ii,None,None],(len(ii),32,64))],1)
def out(ii): return np.stack([(V1[ii]-u1m)/u1s,(V2[ii]-u2m)/u2s],1)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=FNO(n_modes=(12,12),hidden_channels=32,in_channels=4,out_channels=2,n_layers=4).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
train_dl=DataLoader(TensorDataset(torch.tensor(inp(train),dtype=torch.float32),torch.tensor(out(train),dtype=torch.float32),torch.tensor(M[train,None],dtype=torch.float32)),batch_size=8,shuffle=False)
def lossfn(p,t,m): return (((p-t)*m)**2).sum()/m.sum()/2
best=1e99; best_epoch=0; wait=0; hist=[]
for ep in range(1,301):
 model.train(); tl=0
 for a,b,m0 in train_dl: opt.zero_grad(); l=lossfn(model(a.to(device)),b.to(device),m0.to(device)); l.backward(); opt.step(); tl+=l.item()
 model.eval()
 with torch.no_grad(): pv=model(torch.tensor(inp(val),dtype=torch.float32).to(device)); vl=lossfn(pv,torch.tensor(out(val),dtype=torch.float32).to(device),torch.tensor(M[val,None],dtype=torch.float32).to(device)).item()
 hist.append([ep,tl/len(train_dl),vl])
 if vl<best: best=vl; best_epoch=ep; wait=0; torch.save({'model':model.state_dict(),'u1_mean':u1m,'u1_std':u1s,'u2_mean':u2m,'u2_std':u2s},os.path.join(OUT,'best_fno.pt'))
 else: wait+=1
 if wait>=40: break
with open(os.path.join(OUT,'training_history.csv'),'w',newline='') as f: w=csv.writer(f); w.writerow(['epoch','train_loss','val_loss']); w.writerows(hist)
ck=torch.load(os.path.join(OUT,'best_fno.pt'),map_location=device,weights_only=False); model.load_state_dict(ck['model']); model.eval()
metrics=[]; preds={}
with torch.no_grad():
 for i in test:
  pr=model(torch.tensor(inp([i]),dtype=torch.float32).to(device))[0].cpu().numpy(); p1=pr[0]*u1s+u1m; p2=pr[1]*u2s+u2m; e=M[i]>0; metrics.append([i,R[i],np.linalg.norm((p1-V1[i])[e])/np.linalg.norm(V1[i][e]),np.linalg.norm((p2-V2[i])[e])/np.linalg.norm(V2[i][e])]); preds[i]=(p1,p2)
with open(os.path.join(OUT,'test_metrics.csv'),'w',newline='') as f: w=csv.writer(f); w.writerow(['index','R','rel_l2_U1','rel_l2_U2']); w.writerows(metrics)
for i in [test[2]]:
 for j,n in enumerate(['U1','U2']):
  true=[V1,V2][j][i]; pred=preds[i][j]; err=abs(pred-true); e=M[i]>0
  for z,tag in [(true,'true'),(pred,'pred'),(err,'abs_error')]:
   zz=np.ma.masked_where(~e,z); fig,ax=plt.subplots(figsize=(8,4)); lo,hi=true[e].min(),true[e].max(); c=ax.imshow(zz,origin='lower',extent=(0,4,0,2),aspect='equal',vmin=lo if tag!='abs_error' else 0,vmax=hi if tag!='abs_error' else err[e].max()); ax.set_title('FNO '+tag+' '+n); fig.colorbar(c,ax=ax); fig.savefig(os.path.join(OUT,'fno_'+tag+'_'+n+'.png'),dpi=200); plt.close(fig)
arr=np.array(metrics); print('FNO_STATUS = SUCCESS'); print('DEVICE =',device); print('BEST_EPOCH =',best_epoch); print('BEST_VAL_LOSS =',best); print('TRAIN_CASES =',len(train)); print('VAL_CASES =',len(val)); print('TEST_CASES =',len(test)); print('CLOSEST_R_TO_025 =',R[np.argmin(abs(R-.25))]); print('MIN_ABS_R_MINUS_025 =',np.min(abs(R-.25))); print('MEAN_REL_L2_U1 =',arr[:,2].mean()); print('MEAN_REL_L2_U2 =',arr[:,3].mean()); print('MAX_REL_L2_U1 =',arr[:,2].max()); print('MAX_REL_L2_U2 =',arr[:,3].max()); print('TEST_CASE_METRICS =',metrics)
