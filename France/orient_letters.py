exec(open(__file__.replace('orient_letters.py','explore.py')).read().split('fig, axes')[0])
from sklearn.decomposition import PCA
a=np.load(root/'outputs/recover_both.npz');ids=a['indices'];l=a['labels'];s=d.iloc[ids]
groups=sorted(set(l)-{-1},key=lambda k:s[l==k].price.mean())
fig,axes=plt.subplots(6,12,figsize=(24,14))
for row,k in enumerate(groups):
 X=StandardScaler().fit_transform(s.loc[l==k,['longitude','latitude','price','availability_365']])
 p=PCA(n_components=2,whiten=True).fit_transform(X)
 for ax,angle in zip(axes[row],range(0,360,30)):
  theta=np.deg2rad(angle);R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);q=p@R.T
  ax.scatter(q[:,0],q[:,1],s=4);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{row+1}: {angle}°',fontsize=8)
fig.tight_layout();fig.savefig(root/'outputs/orient_letters.png',dpi=130)
