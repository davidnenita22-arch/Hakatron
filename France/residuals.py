exec(open(__file__.replace('residuals.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
configs=[('booking',['price','availability_365','review_scores_rating'],[.15,.2,.3]),('prn',['price','number_of_reviews','review_scores_rating'],[.15,.25,.4]),('all',['price','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','review_scores_rating'],[.5,.8,1.2])]
for name,cols,epslist in configs:
 X=StandardScaler().fit_transform(d[cols].fillna(d[cols].median()))
 for eps in epslist:
  initial=DBSCAN(eps=eps,min_samples=15).fit_predict(X)
  mask=initial==-1; ids=np.flatnonzero(mask);XX=StandardScaler().fit_transform(X[mask])
  l=HDBSCAN(min_cluster_size=15,min_samples=5,copy=True).fit_predict(XX)
  counts=pd.Series(l).value_counts();print(name,eps,'noise',len(ids),'groups',counts.to_dict(),flush=True)
  np.savez(root/f'outputs/residual_{name}_{eps}.npz',indices=ids,labels=l)
  groups=[k for k in counts.index if k>=0][:24]
  fig,axes=plt.subplots(len(groups),2,figsize=(9,3*len(groups)),squeeze=False)
  for (ax,bx),k in zip(axes,groups):
   s=d.iloc[ids[l==k]];ax.scatter(s.longitude,s.latitude,s=6);ax.set_aspect(1/np.cos(np.deg2rad(48.86)));ax.set_title(f'{k}: {len(s)}')
   p=PCA(n_components=2).fit_transform(XX[l==k]);bx.scatter(p[:,0],p[:,1],s=6);bx.set_aspect('equal')
  fig.tight_layout();fig.savefig(root/f'outputs/residual_{name}_{eps}.png',dpi=110);plt.close(fig)
