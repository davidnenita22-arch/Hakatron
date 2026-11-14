exec(open(__file__.replace('feature_grid.py','explore.py')).read().split('fig, axes')[0])
from sklearn.decomposition import PCA
for name in ['booking','booking_leaf']:
 l=np.load(root/f'outputs/{name}_labels.npy');groups=sorted(set(l))
 fig,axes=plt.subplots(int(np.ceil(len(groups)/6)),6,figsize=(18,3*np.ceil(len(groups)/6)),squeeze=False)
 X=StandardScaler().fit_transform(d[['price','review_scores_rating','availability_365']])
 for ax,k in zip(axes.flat,groups):
  s=X[l==k];p=PCA(n_components=2).fit_transform(s);ax.scatter(p[:,0],p[:,1],s=3);ax.set_title(f'{k}: {len(s)}');ax.set_aspect('equal')
 for ax in axes.flat[len(groups):]:ax.axis('off')
 fig.tight_layout();fig.savefig(root/f'outputs/features_{name}.png',dpi=110);plt.close(fig)
