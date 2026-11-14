exec(open(__file__.replace('search_clusters.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
configs={
 'geo_price_rating':['latitude','longitude','price','review_scores_rating'],
 'geo_booking':['latitude','longitude','price','review_scores_rating','availability_365'],
 'geo_reviews':['latitude','longitude','price','review_scores_rating','number_of_reviews'],
 'geo_price':['latitude','longitude','price'],
 'booking_leaf':['price','review_scores_rating','availability_365'],
}
for name,cols in configs.items():
 X=StandardScaler().fit_transform(d[cols])
 l=HDBSCAN(min_cluster_size=30,min_samples=8,cluster_selection_method='leaf',copy=True).fit_predict(X)
 counts=pd.Series(l).value_counts();print(name,counts.to_dict(),flush=True)
 groups=[k for k in counts.index if k>=0]
 fig,axes=plt.subplots(int(np.ceil(len(groups)/6)),6,figsize=(18,3*np.ceil(len(groups)/6)),squeeze=False)
 for ax,k in zip(axes.flat,groups):
  s=d[l==k];ax.scatter(s.longitude,s.latitude,s=3);ax.set_title(f'{k}: {len(s)}');ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
 for ax in axes.flat[len(groups):]:ax.axis('off')
 fig.tight_layout();fig.savefig(root/f'outputs/grid_{name}.png',dpi=100);plt.close(fig)
 np.save(root/f'outputs/{name}_labels.npy',l)
