exec(open(__file__.replace('dbscan_levels.py','explore.py')).read().split('fig, axes')[0])
from sklearn.decomposition import PCA
configs={'booking':['price','availability_365','review_scores_rating'],'prn':['price','number_of_reviews','review_scores_rating'],'geo':['latitude','longitude','price','review_scores_rating']}
for name,cols in configs.items():
 X=StandardScaler().fit_transform(d[cols])
 for eps in [.05,.1,.2,.4,.7]:
  l=DBSCAN(eps=eps,min_samples=12).fit_predict(X)
  counts=pd.Series(l).value_counts();groups=[k for k in counts.index if k>=0 and 40<=counts[k]<=3000][:24]
  print(name,eps,'groups',len(groups),flush=True)
  if not groups:continue
  fig,axes=plt.subplots(int(np.ceil(len(groups)/6)),6,figsize=(18,3*np.ceil(len(groups)/6)),squeeze=False)
  for ax,k in zip(axes.flat,groups):
   s=d[l==k];ax.scatter(s.longitude,s.latitude,s=4);ax.set_title(f'{k}: {len(s)}');ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
  for ax in axes.flat[len(groups):]:ax.axis('off')
  fig.tight_layout();fig.savefig(root/f'outputs/dbscan_{name}_{eps}.png',dpi=110);plt.close(fig)
