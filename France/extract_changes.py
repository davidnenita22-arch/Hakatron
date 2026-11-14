exec(open(__file__.replace('extract_changes.py','compare_source.py')).read().split('print(\'Source rows')[0])
s['price']=pd.to_numeric(s.price.str.replace('$','',regex=False).str.replace(',','',regex=False),errors='coerce')
s.host_is_superhost=s.host_is_superhost.map({'t':True,'f':False})
s=s.rename(columns={'neighbourhood_cleansed':'neighbourhood'})
for c in ['latitude','longitude']:s[c]=s[c].round(5)
for c in ['price','review_scores_rating','reviews_per_month']:s[c]=s[c].round(2)
for excluded in [[],['latitude','longitude'],['price','availability_365','review_scores_rating'],['latitude','longitude','price','availability_365','review_scores_rating']]:
 keys=[c for c in d.columns if c!='listing_id' and c not in excluded]
 lookup=s[keys].drop_duplicates()
 joined=d.reset_index().merge(lookup,on=keys,how='left',indicator=True).sort_values('index')
 mask=joined._merge.eq('left_only').to_numpy()
 print('Excluded',excluded,'unmatched',mask.sum(),flush=True)
 name='full' if not excluded else '_'.join(excluded)
 np.save(root/f'outputs/unmatched_{name}.npy',mask)
