exec(open(__file__.replace('final_cluster_search.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
cols=['longitude','latitude','price','availability_365']
X=StandardScaler().fit_transform(d[cols]);truth=np.load(root/'outputs/unmatched_full.npy')
for minpts in [4,8,15,25]:
 l=HDBSCAN(min_cluster_size=60,min_samples=minpts,copy=True).fit_predict(X)
 print('HDBSCAN',minpts,pd.Series(l).value_counts().to_dict(),flush=True)
 np.save(root/f'outputs/full4_hdb_{minpts}.npy',l)
 for k,n in pd.Series(l).value_counts().items():
  if k>=0 and (truth&(l==k)).sum()>30:print('  ',k,n,'matched planted',(truth&(l==k)).sum(),flush=True)
for eps in [.12,.18,.25,.35,.5]:
 l=DBSCAN(eps=eps,min_samples=12).fit_predict(X)
 print('DBSCAN',eps,pd.Series(l).value_counts().head(15).to_dict(),flush=True)
 np.save(root/f'outputs/full4_db_{eps}.npy',l)
 for k,n in pd.Series(l).value_counts().items():
  if k>=0 and (truth&(l==k)).sum()>30:print('  ',k,n,'matched planted',(truth&(l==k)).sum(),flush=True)
