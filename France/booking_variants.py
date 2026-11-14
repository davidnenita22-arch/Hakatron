exec(open(__file__.replace('booking_variants.py','explore.py')).read().split('fig, axes')[0])
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import itertools
cols=['price','minimum_nights','availability_365','accommodates','bedrooms','beds']
for others in itertools.combinations(cols[1:],2):
 name='__'.join(others);features=['price',*others]
 X=StandardScaler().fit_transform(d[features].fillna(d[features].median()))
 l=HDBSCAN(min_cluster_size=25,min_samples=8,copy=True).fit_predict(X)
 counts=pd.Series(l).value_counts();print(name,counts.head(12).to_dict(),flush=True)
 groups=[k for k in counts.index if k>=0 and 40<counts[k]<2000][:18]
 np.save(root/f'outputs/variants_{name}.npy',l)
 if not groups:continue
 fig,axes=plt.subplots(int(np.ceil(len(groups)/6)),6,figsize=(18,3*np.ceil(len(groups)/6)),squeeze=False)
 for ax,k in zip(axes.flat,groups):
  p=PCA(n_components=2).fit_transform(X[l==k]);ax.scatter(p[:,0],p[:,1],s=4);ax.set_title(f'{k}: {len(p)}');ax.set_aspect('equal')
 for ax in axes.flat[len(groups):]:ax.axis('off')
 fig.tight_layout();fig.savefig(root/f'outputs/variants_{name}.png',dpi=110);plt.close(fig)
