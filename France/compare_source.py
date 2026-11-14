from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
root=Path(__file__).resolve().parent
d=pd.read_csv(root/'data/paris_listings.csv')
cols=[c for c in d.columns if c not in ['listing_id','neighbourhood']]
s=pd.read_csv(root/'data/insideairbnb_2026-09-16.csv.gz',usecols=['id','neighbourhood_cleansed',*cols],low_memory=False)
print('Source rows',len(s))
s['price']=pd.to_numeric(s.price.str.replace('$','',regex=False).str.replace(',','',regex=False),errors='coerce')
s.host_is_superhost=s.host_is_superhost.map({'t':True,'f':False})
scale=[111320*np.cos(np.deg2rad(48.86)),111320]
nn=NearestNeighbors(n_neighbors=1).fit(s[['longitude','latitude']].to_numpy()*scale)
dist,idx=nn.kneighbors(d[['longitude','latitude']].to_numpy()*scale)
matched=s.iloc[idx[:,0]].reset_index(drop=True)
print('Distance quantiles m',np.quantile(dist,[0,.25,.5,.75,.9,.99,1]))
print('Source coordinate matches within 1m',int((dist<1).sum()))
for c in cols:
 if pd.api.types.is_numeric_dtype(d[c]):same=np.isclose(d[c],matched[c],equal_nan=True,atol=.015)
 else:same=d[c].fillna('').eq(matched[c].fillna('')).to_numpy()
 print(c,int(same.sum()),'of',len(d))
out=d.copy();out['nearest_source_distance_m']=dist[:,0];out['source_id']=matched.id
out.to_csv(root/'outputs/source_comparison.csv',index=False)
