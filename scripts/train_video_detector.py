"""Second development-only experiment: video-level aggregation avoids max-pooling false alarms."""
import json
import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,Normalizer
from scripts.evaluate import split_records

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'));private=root/'private'
audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'));dev,_=split_records(audit['records'])
xs=[];ys=[]
for sha,g in dev:
    f=np.load(private/'features'/(sha+'.npz'))['features'].astype(np.float32)
    xs.append(np.concatenate([f.mean(0),f.std(0),np.abs(np.diff(f,axis=0)).mean(0)]));ys.append(g['label'])
x=np.stack(xs);y=np.array(ys)
best=None;reports=[]
for scale in ['standard','l2']:
    for c in [.0001,.001,.01,.1,1,10]:
        scores=[]
        for i in range(len(y)):
            m=make_pipeline(StandardScaler() if scale=='standard' else Normalizer(),LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=2026))
            ix=np.arange(len(y))!=i;m.fit(x[ix],y[ix]);scores.append(m.predict_proba(x[i:i+1])[0,1])
        scores=np.array(scores)
        for threshold in np.linspace(.1,.9,17):
            pred=scores>=threshold
            tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));tn=int(np.sum(~pred&(y==0)));fn=int(np.sum(~pred&(y==1)))
            row=dict(scale=scale,C=c,threshold=float(threshold),tp=tp,fp=fp,tn=tn,fn=fn,accuracy=float(np.mean(pred==y)),balanced_accuracy=(tp/13+tn/11)/2,f1=2*tp/max(1,2*tp+fp+fn))
            reports.append(row);rank=(row['balanced_accuracy'],row['f1'])
            if best is None or rank>best[0]:best=(rank,row,scores.tolist())
        print(json.dumps({'done':scale,'C':c,'best':best[1]}),flush=True)
r=best[1]
m=make_pipeline(StandardScaler() if r['scale']=='standard' else Normalizer(),LogisticRegression(C=r['C'],class_weight='balanced',solver='liblinear',max_iter=1000,random_state=2026)).fit(x,y)
joblib.dump({'pipeline':m,'threshold':r['threshold'],'metadata':r,'feature_version':'qwen-video-v1'},root/'models/video-detector.joblib')
(private/'video-detector-cv.json').write_text(json.dumps({'best':r,'search':reports,'method':'Development leave-one-video-out; hyperparameter selection makes this a development estimate.'},indent=2))
print('VIDEO_DETECTOR_TRAINED',json.dumps(r),flush=True)
