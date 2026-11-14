exec(open(__file__.replace('subspace_search.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import itertools,json
cols=['price','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','review_scores_rating','accommodates','bedrooms','beds','calculated_host_listings_count']
values=StandardScaler().fit_transform(d[cols].fillna(d[cols].median()))
found=[];compact=[]
for no,combo in enumerate(itertools.combinations(range(len(cols)),3)):
 X=values[:,combo];name=','.join(cols[i] for i in combo)
 l=HDBSCAN(min_cluster_size=20,min_samples=6,copy=True).fit_predict(X)
 for k,n in pd.Series(l).value_counts().items():
  if k<0 or not 65<=n<=4000:continue
  ids=np.flatnonzero(l==k);s=d.iloc[ids]
  p=PCA(n_components=2).fit_transform(X[ids]);p/=np.std(p,axis=0)+1e-10
  try:
   dst=NearestNeighbors(n_neighbors=5).fit(p).kneighbors()[0][:,-1]
   score=ConvexHull(p).volume/(n*np.median(dst)**2)
   if np.isfinite(score):found.append((float(score),name,ids.tolist()))
  except Exception:pass
  cs=s.latitude.std()*s.longitude.std()/(d.latitude.std()*d.longitude.std())
  compact.append((float(cs),name,ids.tolist()))
 if no%10==0:print('Combinations processed',no+1,flush=True)
found.sort(reverse=True);compact.sort()
(root/'outputs/subspace_candidates.json').write_text(json.dumps({'shape':found[:100],'compact':compact[:100]}))
for mode,items in [('shape',found[:36]),('compact',compact[:36])]:
 fig,axes=plt.subplots(6,6,figsize=(21,21))
 for ax,(score,name,ids) in zip(axes.flat,items):
  X=values[ids][:,[cols.index(c) for c in name.split(',')]]
  if mode=='shape':p=PCA(n_components=2).fit_transform(X)
  else:p=d.iloc[ids][['longitude','latitude']].to_numpy()*[np.cos(np.deg2rad(48.86)),1]
  ax.scatter(p[:,0],p[:,1],s=3);ax.set_title(f'{score:.3g}, n={len(ids)}\n{name}',fontsize=7);ax.set_aspect('equal')
 fig.tight_layout();fig.savefig(root/f'outputs/subspace_{mode}.png',dpi=120);plt.close(fig)
