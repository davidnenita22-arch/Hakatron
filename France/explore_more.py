exec(open(__file__.replace('explore_more.py','explore.py')).read().split('fig, axes')[0])
cols=['price','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','review_scores_rating','accommodates','calculated_host_listings_count']
fig, axes=plt.subplots(len(cols),len(cols),figsize=(24,24))
for i,y in enumerate(cols):
 for j,x in enumerate(cols):
  ax=axes[i,j]
  if i==j: ax.hist(d[x],bins=80)
  else: ax.scatter(d[x],d[y],s=.2,alpha=.3,rasterized=True)
  if i==len(cols)-1: ax.set_xlabel(x,fontsize=7)
  if j==0: ax.set_ylabel(y,fontsize=7)
  ax.tick_params(labelsize=5)
fig.tight_layout(); fig.savefig(root/'outputs/pairs.png',dpi=120)
cols=['price','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','review_scores_rating']
X=StandardScaler().fit_transform(d[cols].fillna(d[cols].median()))
for eps in [.15,.25,.35,.5]:
 l=DBSCAN(eps=eps,min_samples=10).fit_predict(X)
 print(eps,pd.Series(l).value_counts().to_dict(),flush=True)
 fig,ax=plt.subplots(figsize=(10,7)); ax.scatter(d.longitude,d.latitude,c=l,s=2,cmap='tab20'); ax.set_title(str(eps)); fig.savefig(root/f'outputs/cluster_{eps}.png'); plt.close(fig)
