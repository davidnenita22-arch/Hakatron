exec(open(__file__.replace('grid_search.py','explore.py')).read().split('fig, axes')[0])
import json
candidates=[]
for path in (root/'outputs').glob('compact_*.npy'):
 l=np.load(path)
 for k in set(l)-{-1}:
  ids=np.flatnonzero(l==k);s=d.iloc[ids]
  if not 10<=len(s)<=100 or s.latitude.std()<.002 or s.longitude.std()<.004:continue
  p=s[['longitude','latitude']].to_numpy();p=(p-p.min(axis=0))/np.ptp(p,axis=0)
  best=[]
  for col in p.T:
   best.append(min(np.mean(np.abs(col*n-np.round(col*n))) for n in range(3,13)))
  # Letters contain several aligned points along their vertical and horizontal strokes.
  dif=np.abs(p[:,None,:]-p[None,:,:]);diag=~np.eye(len(p),dtype=bool)
  alignment=np.mean(np.any(dif<.018,axis=2)[diag])
  score=sum(best)-alignment*.25
  candidates.append((float(score),path.stem,int(k),ids.tolist()))
candidates.sort()
(root/'outputs/grid_candidates.json').write_text(json.dumps(candidates))
fig,axes=plt.subplots(6,6,figsize=(21,21))
for ax,(score,name,k,ids) in zip(axes.flat,candidates[:36]):
 s=d.iloc[ids];ax.scatter(s.longitude,s.latitude,s=12);ax.set_title(f'{score:.3f}, {name} {k}: {len(s)}',fontsize=8);ax.set_aspect(1/np.cos(np.deg2rad(48.86)))
fig.tight_layout();fig.savefig(root/'outputs/grid_candidates.png',dpi=120)
