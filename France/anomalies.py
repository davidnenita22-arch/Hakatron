exec(open(__file__.replace('anomalies.py','explore.py')).read().split('fig, axes')[0])
from sklearn.ensemble import IsolationForest
configs={'booking':['price','minimum_nights','availability_365','number_of_reviews','reviews_per_month','review_scores_rating'],'all':['latitude','longitude','price','minimum_nights','availability_365','number_of_reviews','number_of_reviews_ltm','reviews_per_month','review_scores_rating','accommodates','bedrooms','beds','calculated_host_listings_count']}
fig,axes=plt.subplots(2,4,figsize=(20,10))
for row,(name,cols) in enumerate(configs.items()):
 X=d[cols].fillna(d[cols].median());score=-IsolationForest(n_estimators=300,random_state=42).fit(X).score_samples(X)
 for ax,n in zip(axes[row],[100,300,1000,3000]):
  ids=np.argsort(score)[-n:];s=d.iloc[ids];ax.scatter(s.longitude,s.latitude,c=s.price,s=5,cmap='turbo',vmin=40,vmax=600);ax.set_title(f'{name} top {n}');ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
 np.save(root/f'outputs/anomaly_{name}.npy',score)
fig.tight_layout();fig.savefig(root/'outputs/anomalies.png',dpi=120)
