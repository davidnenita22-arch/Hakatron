exec(open(__file__.replace('small_clusters.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
for name,cols in [('prn',['price','review_scores_rating','number_of_reviews']),('prm',['price','review_scores_rating','minimum_nights']),('prv',['price','review_scores_rating','reviews_per_month']),('pav',['price','availability_365','number_of_reviews']),('all',['price','review_scores_rating','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','accommodates','bedrooms','beds','calculated_host_listings_count'])]:
 X=StandardScaler().fit_transform(d[cols].fillna(d[cols].median()))
 l=HDBSCAN(min_cluster_size=8,min_samples=4,copy=True).fit_predict(X)
 counts=pd.Series(l).value_counts();print(name,counts.head(15).to_dict(),flush=True)
 groups=[k for k in sorted(set(l)) if k>=0 and 45<=counts[k]<3000]
 fig,axes=plt.subplots(int(np.ceil(len(groups)/8)),8,figsize=(24,3*np.ceil(len(groups)/8)),squeeze=False)
 for ax,k in zip(axes.flat,groups):
  s=X[l==k];p=PCA(n_components=2).fit_transform(s);ax.scatter(p[:,0],p[:,1],s=5);ax.set_title(f'{k}: {len(s)}');ax.set_aspect('equal')
 for ax in axes.flat[len(groups):]:ax.axis('off')
 fig.tight_layout();fig.savefig(root/f'outputs/small_{name}.png',dpi=100);plt.close(fig)
 np.save(root/f'outputs/small_{name}.npy',l)
