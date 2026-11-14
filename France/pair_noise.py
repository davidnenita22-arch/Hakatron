exec(open(__file__.replace('pair_noise.py','explore.py')).read().split('fig, axes')[0])
import itertools
cols=['price','review_scores_rating','availability_365','number_of_reviews','reviews_per_month','minimum_nights','number_of_reviews_ltm']
fig,axes=plt.subplots(4,6,figsize=(24,16))
for ax,pair in zip(axes.flat,itertools.combinations(cols,2)):
 X=StandardScaler().fit_transform(d[list(pair)].fillna(d[list(pair)].median()))
 candidates=[]
 for eps in [.03,.05,.075,.1,.15,.25,.4]:
  l=DBSCAN(eps=eps,min_samples=20).fit_predict(X)
  mask=l==-1;candidates.append((abs(mask.sum()-1500),eps,mask))
 _,eps,mask=min(candidates,key=lambda x:x[0])
 s=X[mask];ax.scatter(s[:,0],s[:,1],s=6);ax.set_title(f'{pair}\neps={eps}, noise={len(s)}',fontsize=9)
 np.save(root/f'outputs/pairnoise_{pair[0]}_{pair[1]}.npy',mask)
for ax in axes.flat[21:]:ax.axis('off')
fig.tight_layout();fig.savefig(root/'outputs/pair_noise.png',dpi=120)
