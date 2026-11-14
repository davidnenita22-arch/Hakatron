exec(open(__file__.replace('geo_check.py','explore.py')).read().split('fig, axes')[0])
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
coords=d[['longitude','latitude']].to_numpy()*[np.cos(np.deg2rad(48.86)),1]
idx=NearestNeighbors(n_neighbors=51).fit(coords).kneighbors(return_distance=False)
y=LabelEncoder().fit_transform(d.neighbourhood)
probs=np.array([np.bincount(y[ids],minlength=20)/len(ids) for ids in idx])
own=probs[np.arange(len(d)),y]
fig,axes=plt.subplots(1,3,figsize=(18,7))
for ax,t in zip(axes,[.05,.2,.5]):
 mask=own<t;s=d[mask];ax.scatter(s.longitude,s.latitude,s=5,c=s.price,cmap='turbo');ax.set_title(f'Own neighbourhood frequency < {t}: {len(s)}');ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
fig.tight_layout();fig.savefig(root/'outputs/geo_check.png',dpi=130)
d.assign(own_neighbourhood_frequency=own).to_csv(root/'outputs/geo_check.csv',index=False)
