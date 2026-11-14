exec(open(__file__.replace('recover.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
mask=np.load(root/'outputs/unmatched_full.npy');ids=np.flatnonzero(mask);s=d.iloc[ids]
for name,cols in [('colours',['price','availability_365']),('geo',['longitude','latitude']),('both',['longitude','latitude','price','availability_365'])]:
 X=StandardScaler().fit_transform(s[cols])
 l=HDBSCAN(min_cluster_size=40,min_samples=6,copy=True).fit_predict(X)
 print(name,pd.Series(l).value_counts().to_dict())
 np.savez(root/f'outputs/recover_{name}.npz',indices=ids,labels=l)
 groups=sorted(set(l)-{-1},key=lambda k:s[l==k].price.mean())
 fig,axes=plt.subplots(len(groups),3,figsize=(13,3.5*len(groups)),squeeze=False)
 for (ax,bx,cx),k in zip(axes,groups):
  ss=s[l==k];ax.scatter(ss.longitude,ss.latitude,s=8);ax.set_aspect(1/np.cos(np.deg2rad(48.86)));ax.set_title(f'{k}: {len(ss)}, price {ss.price.mean():.1f}')
  bx.scatter(ss.price,ss.availability_365,s=8)
  p=PCA(n_components=2).fit_transform(X[l==k]);cx.scatter(p[:,0],p[:,1],s=8);cx.set_aspect('equal')
 fig.tight_layout();fig.savefig(root/f'outputs/recover_{name}.png',dpi=120);plt.close(fig)
