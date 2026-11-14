"""Conditional specialists. All feature/branch selection is fitted on training rows."""
import numpy as np
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

def cat(iterations=700,depth=4):
 return CatBoostClassifier(iterations=iterations,depth=depth,learning_rate=.05,verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)
class BranchExperts:
 def _fit_branch(self,X,y,gate,parity):
  root=cat(600).fit(X[:,parity],y)
  agree=(root.predict(X[:,parity]).ravel().astype(int)==y).astype(int)
  stump=DecisionTreeClassifier(max_depth=1,min_samples_leaf=max(30,int(.1*len(y))),random_state=42).fit(X[:,[gate]],agree)
  threshold=float(stump.tree_.threshold[0]);below=X[:,gate]<=threshold
  if below.sum()==0 or (~below).sum()==0:raise ValueError('No branch split')
  parity_below=agree[below].mean()>agree[~below].mean();region=below if parity_below else ~below
  parity_model=cat(1000).fit(X[region][:,parity],y[region]);linear=LogisticRegression(C=1,max_iter=3000).fit(X[~region],y[~region])
  return {'gate':int(gate),'parity':list(map(int,parity)),'threshold':threshold,'parity_below':bool(parity_below),'parity_model':parity_model,'linear':linear,'parity_count':int(region.sum())}
 def _predict(self,X,branch):
  gate=branch['gate'];below=X[:,gate]<=branch['threshold'];sel=below if branch['parity_below'] else ~below
  p=np.zeros(len(X))
  if sel.any():p[sel]=branch['parity_model'].predict_proba(X[sel][:,branch['parity']])[:,1]
  if (~sel).any():p[~sel]=branch['linear'].predict_proba(X[~sel])[:,1]
  return p
 def fit(self,X,y):
  selector=cat(1000,5).fit(X,y);self.top4=np.argsort(selector.feature_importances_)[-4:]
  tr,va=train_test_split(np.arange(len(y)),test_size=.2,random_state=73,stratify=y);choices=[]
  for gate in self.top4:
   parity=self.top4[self.top4!=gate];branch=self._fit_branch(X[tr],y[tr],gate,parity);p=self._predict(X[va],branch);score=accuracy_score(y[va],p>=.5);choices.append((float(score),int(gate),parity.tolist()))
  self.choices=choices;_,gate,parity=max(choices,key=lambda t:t[0]);self.branch=self._fit_branch(X,y,gate,parity);return self
 def predict_proba(self,X):return self._predict(X,self.branch)
 def metadata(self):
  b=self.branch
  return {'top4_channels':[f'c{i+1:02}' for i in self.top4],'gate_channel':f'c{b["gate"]+1:02}','gate_threshold_standardized':b['threshold'],'parity_channels':[f'c{i+1:02}' for i in b['parity']],'parity_below_gate':b['parity_below'],'inner_selection_scores':[{'score':s,'gate':f'c{g+1:02}'} for s,g,p in self.choices],'parity_count':b['parity_count']}
class MixtureExperts:
 def fit(self,X,y):
  self.scaler=StandardScaler().fit(X);Z=self.scaler.transform(X);self.signal=abs(kurtosis(Z))<.3
  self.gate=GaussianMixture(2,covariance_type='diag',n_init=5,random_state=42).fit(Z[:,~self.signal]);self.narrow=int(np.argmin(self.gate.covariances_.mean(1)));sel=self.gate.predict(Z[:,~self.signal])==self.narrow
  self.small=[make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),LogisticRegression(C=.1,max_iter=3000)),SVC(C=3,gamma=.03,probability=True,random_state=42)]
  for m in self.small:m.fit(Z[sel][:,self.signal],y[sel])
  self.large=BranchExperts().fit(Z[~sel],y[~sel]);self.regime_counts=[int(sel.sum()),int((~sel).sum())];return self
 def predict_proba(self,X):
  Z=self.scaler.transform(X);g=self.gate.predict_proba(Z[:,~self.signal])[:,self.narrow];p=np.mean([m.predict_proba(Z[:,self.signal])[:,1] for m in self.small],axis=0);return g*p+(1-g)*self.large.predict_proba(Z)
 def metadata(self):
  b=self.large.branch;raw=self.scaler.mean_[b['gate']]+b['threshold']*self.scaler.scale_[b['gate']]
  return {'gaussian_channels':[f'c{i+1:02}' for i in np.flatnonzero(self.signal)],'regime_counts_narrow_broad':self.regime_counts,'gate_mean_variances':self.gate.covariances_.mean(1).tolist(),'broad_branch':self.large.metadata(),'broad_gate_threshold_raw':float(raw)}
