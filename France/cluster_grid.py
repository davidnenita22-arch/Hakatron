exec(open(__file__.replace('cluster_grid.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
import itertools
cols=['price','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','review_scores_rating','accommodates','bedrooms','beds','calculated_host_listings_count','latitude','longitude']
for name,features in [('booking',['price','availability_365','review_scores_rating']),('reviews',['price','reviews_per_month','review_scores_rating']),('all',cols),('geo',['latitude','longitude'])]:
 X=StandardScaler().fit_transform(d[features].fillna(d[features].median()))
 l=HDBSCAN(min_cluster_size=20,min_samples=5).fit_predict(X)
 counts=pd.Series(l).value_counts(); print(name,counts.to_dict(),flush=True)
 groups=[k for k in counts.index if k>=0 and counts[k]<5000]
 fig,axes=plt.subplots(int(np.ceil(len(groups)/6)),6,figsize=(18,3*np.ceil(len(groups)/6)),squeeze=False)
 for ax,k in zip(axes.flat,groups):
  s=d[l==k]; ax.scatter(s.longitude,s.latitude,s=4); ax.set_title(f'{k}: {len(s)}'); ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
 for ax in axes.flat[len(groups):]: ax.axis('off')
 fig.tight_layout();fig.savefig(root/f'outputs/grid_{name}.png',dpi=120);plt.close(fig)
 np.save(root/f'outputs/{name}_labels.npy',l)
