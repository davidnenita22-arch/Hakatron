exec(open(__file__.replace('projections.py','explore.py')).read().split('fig, axes')[0])
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
cols=['price','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','review_scores_rating','accommodates','bedrooms','beds','calculated_host_listings_count']
fig,axes=plt.subplots(3,5,figsize=(22,13))
for row,mode in enumerate(['standard','robust','log']):
 a=d[cols].fillna(d[cols].median()).copy()
 if mode=='log':
  for c in cols:
   if c not in ['review_scores_rating','availability_365']:a[c]=np.log1p(a[c])
 X=(RobustScaler() if mode=='robust' else StandardScaler()).fit_transform(a)
 p=PCA().fit_transform(X)
 for ax,(i,j) in zip(axes[row],[(0,1),(1,2),(2,3),(3,4),(4,5)]):
  ax.scatter(p[:,i],p[:,j],s=1,alpha=.4);ax.set_title(f'{mode}: PC{i+1} / PC{j+1}')
fig.tight_layout();fig.savefig(root/'outputs/projections.png',dpi=140)
