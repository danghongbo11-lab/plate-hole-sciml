import os, numpy as np, gmsh, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from skfem import MeshTri, Basis, ElementTriP1, ElementVector, asm, condense, solve
from skfem.models.elasticity import linear_elasticity
ROOT=os.path.dirname(os.path.abspath(__file__)); L=4.; H=2.; CX=2.; CY=1.; R=float(os.environ.get('PYFEM_R','.25')); E=1.; nu=.3
C=E/(1-nu**2)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
gmsh.initialize(); gmsh.model.add('plate'); ps=[gmsh.model.geo.addPoint(x,y,0,.08) for x,y in [(0,0),(L,0),(L,H),(0,H)]]; cl=gmsh.model.geo.addPoint(CX,CY,0,.015); q=gmsh.model.geo.addPoint(CX+R,CY,0,.015); a=gmsh.model.geo.addPoint(CX,CY+R,0,.015); b=gmsh.model.geo.addPoint(CX-R,CY,0,.015); c=gmsh.model.geo.addPoint(CX,CY-R,0,.015)
ls=[gmsh.model.geo.addLine(ps[i],ps[(i+1)%4]) for i in range(4)]; arcs=[gmsh.model.geo.addCircleArc(q,cl,a),gmsh.model.geo.addCircleArc(a,cl,b),gmsh.model.geo.addCircleArc(b,cl,c),gmsh.model.geo.addCircleArc(c,cl,q)]; gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(ls),gmsh.model.geo.addCurveLoop(arcs)]); gmsh.model.geo.synchronize(); gmsh.model.mesh.setSize(gmsh.model.getEntities(0),.08); gmsh.model.mesh.setSize([(2,q),(2,a),(2,b),(2,c)],.015); gmsh.model.mesh.generate(2)
nodes=gmsh.model.mesh.getNodes(); pts=nodes[1].reshape(-1,3)[:,:2]; ids={int(v):i for i,v in enumerate(nodes[0])}; _,_,en=gmsh.model.mesh.getElements(2); conn=np.array([[ids[int(v)] for v in en[0][3*i:3*i+3]] for i in range(len(en[0])//3)],int).T; gmsh.finalize()
m=MeshTri(pts.T,conn).remove_unused_nodes(); pts=m.p.T; conn=m.t; basis=Basis(m,ElementVector(ElementTriP1()))
lam=E*nu/(1-nu**2); mu=E/(2*(1+nu)); K=asm(linear_elasticity(Lambda=lam,Mu=mu),basis); f=np.zeros(basis.N); rn=np.where(np.isclose(pts[:,0],L))[0]; f[2*rn]=H/len(rn)
left_dofs=basis.get_dofs(lambda x: np.isclose(x[0],0.0)); D_left_x=np.asarray(left_dofs.nodal['u^1']).reshape(-1)
dist2=pts[:,0]**2+pts[:,1]**2; anchor_node=int(np.argmin(dist2)); anchor_xy=pts[anchor_node]; D_anchor_y=int(basis.nodal_dofs[1,anchor_node])
D=np.unique(np.concatenate([D_left_x,np.asarray([D_anchor_y]).reshape(-1)]))
print('LEFT_U1_DOF_COUNT =',len(D_left_x)); print('ANCHOR_NODE =',anchor_node); print('ANCHOR_XY =',anchor_xy); print('ANCHOR_U2_DOF =',D_anchor_y); print('TOTAL_DIRICHLET_DOF_COUNT =',len(D)); print('K_SHAPE =',K.shape)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
aa=np.r_[conn[0],conn[1],conn[1],conn[2],conn[2],conn[0]]; bb=np.r_[conn[1],conn[0],conn[2],conn[1],conn[0],conn[2]]; graph=coo_matrix((np.ones(len(aa)),(aa,bb)),shape=(len(pts),len(pts))).tocsr()
print('MESH_CONNECTED_COMPONENTS =',connected_components(graph,directed=False,return_labels=False))
if len(D)<=len(D_left_x) or abs(anchor_xy[0])>1e-6 or abs(anchor_xy[1])>1e-6: raise RuntimeError('Boundary DOF construction failed')
u=solve(*condense(K,f,D=D)); 
if not np.all(np.isfinite(u)): raise RuntimeError('Non-finite displacement after solve')
uv=np.vstack((u[0::2],u[1::2]))
cent=[]; ss=[]
for tri in conn.T:
 X=pts[tri]; A=np.array([[X[1,0]-X[0,0],X[2,0]-X[0,0]],[X[1,1]-X[0,1],X[2,1]-X[0,1]]]); g=np.linalg.solve(A,np.array([[uv[0,tri[1]]-uv[0,tri[0]],uv[0,tri[2]]-uv[0,tri[0]]],[uv[1,tri[1]]-uv[1,tri[0]],uv[1,tri[2]]-uv[1,tri[0]]]]).T); ss.append(C@np.array([g[0,0],g[1,1],g[0,1]+g[1,0]])); cent.append(X.mean(0))
cent=np.array(cent); ss=np.array(ss); np.savetxt(os.path.join(ROOT,'nodal_displacement.csv'),np.c_[pts,uv.T],delimiter=',',header='x,y,U1,U2',comments=''); np.savetxt(os.path.join(ROOT,'stress_results.csv'),np.c_[cent,ss],delimiter=',',header='x,y,S11,S22,S12',comments='')
tri=mtri.Triangulation(pts[:,0],pts[:,1],conn.T); tpts=pts[conn.T]; tri.set_mask(np.any(((tpts[:,:,0]-CX)**2+(tpts[:,:,1]-CY)**2)<R**2,axis=1))
for n,z,tit in [('U1',uv[0],'FEM displacement U1'),('U2',uv[1],'FEM displacement U2')]:
 fig,ax=plt.subplots(figsize=(8,4)); zc=ax.tricontourf(tri,z,56); ax.set(xlim=(0,4),ylim=(0,2),aspect='equal',title=tit); fig.colorbar(zc,ax=ax); fig.savefig(os.path.join(ROOT,'fem_'+n+'.png'),dpi=220); plt.close(fig)
for j,n in enumerate(['S11','S22','S12']):
 fig,ax=plt.subplots(figsize=(8,4)); zc=ax.scatter(cent[:,0],cent[:,1],c=ss[:,j],s=5); ax.set(xlim=(0,4),ylim=(0,2),aspect='equal',title='FEM stress '+n); fig.colorbar(zc,ax=ax); fig.savefig(os.path.join(ROOT,'fem_'+n+'.png'),dpi=220); plt.close(fig)
iu=np.argmax(uv[0]); is1=np.argmax(ss[:,0]); print('MAX_U1 =',uv[0,iu]); print('LOCATION_MAX_U1 =',pts[iu]); print('U1_REL_ERROR =',abs(uv[0,iu]-4.335395)/4.335395); print('MAX_ABS_U2 =',max(abs(uv[1]))); print('LOCATION_MAX_ABS_U2 =',pts[np.argmax(abs(uv[1]))]); print('MAX_S11 =',max(ss[:,0])); print('LOCATION_MAX_S11 =',cent[is1]); print('S11_REL_ERROR =',abs(max(ss[:,0])-3.0462396)/3.0462396); print('MAX_ABS_S22 =',max(abs(ss[:,1]))); print('LOCATION_MAX_ABS_S22 =',cent[np.argmax(abs(ss[:,1]))]); print('MAX_ABS_S12 =',max(abs(ss[:,2]))); print('LOCATION_MAX_ABS_S12 =',cent[np.argmax(abs(ss[:,2]))]); print('NODE_COUNT =',len(pts)); print('ELEMENT_COUNT =',len(conn.T)); print('STRESS_LOCATION = element_center')
