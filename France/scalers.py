exec(open(__file__.replace('scalers.py','explore.py')).read().split('fig, axes')[0])
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
configs={'booking':['price','availability_365','review_scores_rating'],'prn':['price','number_of_reviews','review_scores_rating'],'prv':['price','reviews_per_month','review_scores_rating'],'prm':['price','minimum_nights','review_scores_rating'],'all':['price','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','review_scores_rating','accommodates','bedrooms','beds','calculated_host_listings_count']}
for name,cols in configs.items():
 for mode,scaler in [('minmax',MinMaxScaler()),('robust',RobustScaler()),('log',StandardScaler())]:
  a=d[cols].fillna(d[cols].median()).copy()
  if mode=='log':
   for c in cols:
    if c not in ['review_scores_rating','availability_365']:a[c]=np.log1p(a[c])
  X=scaler.fit_transform(a)
  l=HDBSCAN(min_cluster_size=25,min_samples=8,copy=True).fit_predict(X)
  counts=pd.Series(l).value_counts();print(name,mode,counts.head(20).to_dict(),flush=True)
  np.save(root/f'outputs/scaler_{name}_{mode}.npy',l)
  groups=[k for k in counts.index if k>=0 and counts[k]<10000][:18]
  if not groups:continue
  fig,axes=plt.subplots(len(groups),2,figsize=(8,3*len(groups)),squeeze=False)
  for (ax,bx),k in zip(axes,groups):
   s=d[l==k];ax.scatter(s.longitude,s.latitude,s=5);ax.set_aspect(1/np.cos(np.deg2rad(48.86)));ax.set_title(f'{k}: {len(s)}')
   p=PCA(n_components=2).fit_transform(X[l==k]);bx.scatter(p[:,0],p[:,1],s=5);bx.set_aspect('equal')
  fig.tight_layout();fig.savefig(root/f'outputs/scaler_{name}_{mode}.png',dpi=100);plt.close(fig)
