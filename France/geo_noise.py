exec(open(__file__.replace('geo_noise.py','explore.py')).read().split('fig, axes')[0])
coords=d[['longitude','latitude']].to_numpy()*[111320*np.cos(np.deg2rad(48.86)),111320]
fig,axes=plt.subplots(3,5,figsize=(22,14))
for row,minpts in enumerate([5,12,25]):
 for ax,radius in zip(axes[row],[100,150,200,250,300]):
  l=DBSCAN(eps=radius,min_samples=minpts).fit_predict(coords)
  s=d[l==-1];ax.scatter(s.longitude,s.latitude,c=s.price,s=5,cmap='turbo',vmin=40,vmax=600);ax.set_title(f'r={radius}, min={minpts}, noise={len(s)}');ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
  np.save(root/f'outputs/geonoise_{radius}_{minpts}.npy',l)
fig.tight_layout();fig.savefig(root/'outputs/geo_noise.png',dpi=120)
