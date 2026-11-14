from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/point-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

root = Path(__file__).resolve().parent
d = pd.read_csv(root/'data/paris_listings.csv')
features = ['price', 'availability_365', 'review_scores_rating']
fig, axes = plt.subplots(2, 3, figsize=(16,9))
for ax, col in zip(axes[0], features):
    ax.scatter(d.longitude,d.latitude,c=d[col],s=1,cmap='turbo')
    ax.set_title(col)
for ax, (x,y) in zip(axes[1], [('price','availability_365'),('price','review_scores_rating'),('availability_365','review_scores_rating')]):
    ax.scatter(d[x],d[y],s=1,alpha=.3)
    ax.set_xlabel(x); ax.set_ylabel(y)
fig.tight_layout(); fig.savefig(root/'outputs/exploration.png',dpi=150)
X = StandardScaler().fit_transform(d[features])
for eps in [.03,.05,.075,.1,.15,.2]:
    labels = DBSCAN(eps=eps,min_samples=15).fit_predict(X)
    counts = pd.Series(labels).value_counts()
    print(eps, counts.to_dict())
