exec(open(__file__.replace('noise.py','explore.py')).read().split('fig, axes')[0])
fig,axes=plt.subplots(2,4,figsize=(20,10))
for row,name in enumerate(['booking','all']):
 l=np.load(root/f'outputs/{name}_labels.npy'); s=d[l==-1]
 for ax,(x,y) in zip(axes[row],[('longitude','latitude'),('price','review_scores_rating'),('price','availability_365'),('price','number_of_reviews')]):
  ax.scatter(s[x],s[y],s=4);ax.set_xlabel(x);ax.set_ylabel(y);ax.set_title(name+' noise')
fig.tight_layout();fig.savefig(root/'outputs/noise.png',dpi=150)
