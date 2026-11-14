"""Reproducible supervised mixture-of-specialists solution.
Usage: .venv/bin/python solve.py [--skip-cv]
The vault is never read during model fitting or validation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,json,hashlib,time,warnings,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss
warnings.filterwarnings('ignore',category=FutureWarning)
HERE=Path(__file__).resolve().parent
CHANNELS=[f'c{i:02}' for i in range(1,49)]
from structural_model import MixtureExperts

def metrics(y,p):
 return {'accuracy':float(accuracy_score(y,p>=.5)),'roc_auc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p)),'errors':int(np.sum((p>=.5)!=y)),'n':len(y)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--skip-cv',action='store_true');args=ap.parse_args();out=HERE/'outputs';out.mkdir(exist_ok=True)
 train=pd.read_csv(HERE/'dataset/train.csv');sealed=pd.read_csv(HERE/'dataset/sealed.csv');X=train[CHANNELS].to_numpy();y=train.vrancea.to_numpy();S=sealed[CHANNELS].to_numpy()
 report={'seed':0,'folds':5,'cv':'StratifiedKFold(5,shuffle=True,random_state=0)','training_shape':list(train.shape),'sealed_shape':list(sealed.shape),'input_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'dataset/train.csv',HERE/'dataset/sealed.csv']}}
 if not args.skip_cv:
  cv=StratifiedKFold(5,shuffle=True,random_state=0);oof=np.zeros(len(y));baseline={name:[] for name in ['logistic_regression','rbf_svm']};folds=[]
  for fold,(tr,va) in enumerate(cv.split(X,y),1):
   t=time.time();m=MixtureExperts().fit(X[tr],y[tr]);p=m.predict_proba(X[va]);oof[va]=p;f={'fold':fold,**metrics(y[va],p),**m.metadata(),'seconds':time.time()-t};folds.append(f);print('specialists',json.dumps(f),flush=True)
   for name,model in [('logistic_regression',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000))),('rbf_svm',make_pipeline(StandardScaler(),SVC(C=3,gamma='scale')))]:
    model.fit(X[tr],y[tr]);s=float(accuracy_score(y[va],model.predict(X[va])));baseline[name].append(s);print('baseline',fold,name,s,flush=True)
   report.update({'fold_results':folds,'baseline_cv':{name:{'scores':scores,'mean_accuracy':float(np.mean(scores))} for name,scores in baseline.items()},'specialist_oof':metrics(y,oof) if fold==5 else None});(out/'validation.json').write_text(json.dumps(report,indent=2))
  pd.DataFrame({'record_id':train.record_id,'vrancea':y,'oof_probability':oof,'oof_prediction':(oof>=.5).astype(int)}).to_csv(out/'out_of_fold_predictions.csv',index=False)
  print('OOF',report['specialist_oof'],flush=True)
 m=MixtureExperts().fit(X,y);p=m.predict_proba(S);pred=(p>=.5).astype(int)
 pd.DataFrame({'record_id':sealed.record_id,'vrancea':pred}).to_csv(HERE/'predictions.csv',index=False)
 pd.DataFrame({'record_id':sealed.record_id,'vrancea':pred,'probability_vrancea':p,'model_confidence':np.maximum(p,1-p)}).to_csv(out/'sealed_model_probabilities.csv',index=False)
 report['final_model']=m.metadata();report['prediction_counts']={str(k):int(v) for k,v in zip(*np.unique(pred,return_counts=True))};(out/'run_summary.json').write_text(json.dumps(report,indent=2))
 print('FINAL',json.dumps(report['final_model']),flush=True)
 r=subprocess.run([sys.executable,str(HERE/'evaluation_script.py'),str(HERE/'predictions.csv')],capture_output=True,text=True);(out/'evaluator_model.txt').write_text(r.stdout+r.stderr);print('EVALUATOR',r.stdout.strip(),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
if __name__=='__main__':main()
