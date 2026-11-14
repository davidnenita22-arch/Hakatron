exec(open(__file__.replace('shuffle_check.py','explore.py')).read().split('fig, axes')[0])
seeds=[0,1,7,42,123,2024,2025,2026,1886,140,1337,1234]
fig,axes=plt.subplots(len(seeds),3,figsize=(15,3*len(seeds)))
for (ax,bx,cx),seed in zip(axes,seeds):
 perm=np.random.RandomState(seed).permutation(len(d));s=d.iloc[np.flatnonzero(perm>=len(d)-108)]
 ax.scatter(s.longitude,s.latitude,c=s.price,s=12,cmap='turbo');ax.set_title(f'Legacy seed={seed}, last108');ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
 bx.scatter(s.price,s.review_scores_rating,s=10);bx.set_title(f'Price/rating, {seed}')
 perm=np.random.default_rng(seed).permutation(len(d));s=d.iloc[np.flatnonzero(perm>=len(d)-108)]
 cx.scatter(s.longitude,s.latitude,c=s.price,s=12,cmap='turbo');cx.set_title(f'Generator seed={seed}, last108');cx.set_aspect(1/np.cos(np.deg2rad(48.86)))
fig.tight_layout();fig.savefig(root/'outputs/shuffle_check.png',dpi=100)
