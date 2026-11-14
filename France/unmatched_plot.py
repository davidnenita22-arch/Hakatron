exec(open(__file__.replace('unmatched_plot.py','explore.py')).read().split('fig, axes')[0])
mask=np.load(root/'outputs/unmatched_full.npy');s=d[mask]
print(s.describe().to_string())
fig,axes=plt.subplots(2,3,figsize=(16,10))
for ax,col in zip(axes[0],['price','availability_365','review_scores_rating']):
 ax.scatter(s.longitude,s.latitude,c=s[col],s=9,cmap='turbo');ax.set_title(col);ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
for ax,(x,y) in zip(axes[1],[('price','review_scores_rating'),('price','availability_365'),('price','number_of_reviews')]):
 ax.scatter(s[x],s[y],s=6);ax.set_xlabel(x);ax.set_ylabel(y)
fig.tight_layout();fig.savefig(root/'outputs/unmatched.png',dpi=140)
s.to_csv(root/'outputs/unmatched.csv',index=False)
