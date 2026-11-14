exec(open(__file__.replace('compact_clusters.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
configs={'booking':['price','availability_365','review_scores_rating'],'prn':['price','number_of_reviews','review_scores_rating'],'prv':['price','reviews_per_month','review_scores_rating'],'prm':['price','minimum_nights','review_scores_rating'],'pa':['price','availability_365'],'pr':['price','review_scores_rating']}
for name,cols in configs.items():
 X=StandardScaler().fit_transform(d[cols].fillna(d[cols].median()))
 l=HDBSCAN(min_cluster_size=8,min_samples=3,copy=True).fit_predict(X)
 np.save(root/f'outputs/compact_{name}.npy',l)
 groups=[]
 for k in set(l)-{-1}:
  s=d[l==k]
  if s.latitude.std()<.008 and s.longitude.std()<.015:groups.append(k)
 print(name,'total',len(set(l))-1,'compact',len(groups),flush=True)
 if not groups:continue
 fig,axes=plt.subplots(int(np.ceil(len(groups)/6)),6,figsize=(18,3*np.ceil(len(groups)/6)),squeeze=False)
 for ax,k in zip(axes.flat,groups):
  s=d[l==k];ax.scatter(s.longitude,s.latitude,s=9);ax.set_title(f'{k}: {len(s)}');ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
 for ax in axes.flat[len(groups):]:ax.axis('off')
 fig.tight_layout();fig.savefig(root/f'outputs/compact_{name}.png',dpi=110);plt.close(fig)
