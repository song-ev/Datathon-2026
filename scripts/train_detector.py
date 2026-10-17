"""Train/tune only on development groups. All video-derived features stay local."""
import json
import os
import re
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forte.engine import Engine
from forte.features import extract,temporal
from scripts.evaluate import split_records

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
private=root/'private';cache=private/'features';cache.mkdir(exist_ok=True)
audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'))
dev,_=split_records(audit['records'])
paths={p.name:p for p in (root/'data').rglob('*.mov')}
engine=Engine();rows=[]
for sha,group in dev:
    started=time.monotonic();file=cache/(sha+'.npz')
    if file.exists():
        loaded=np.load(file);ts,base=loaded['times'],loaded['features']
    else:
        ts,base=extract(engine,paths[group['records'][0]['name']])
        np.savez_compressed(file,times=ts,features=base.astype(np.float16))
    targets=[]
    for r in group['records']:
        text=str(r.get('raw_time') or '');nums=re.findall(r'\d+',text)
        if nums:targets.append(float(nums[0])*60+float(nums[1]) if 'мин' in text and len(nums)>1 else float(nums[0]))
    x=temporal(base.astype(np.float32))
    y=np.zeros(len(ts),dtype=int)
    mask=np.ones(len(ts),dtype=bool)
    if group['label']:
        distance=np.min(np.abs(ts[:,None]-np.array(targets)[None,:]),axis=1)
        mask=distance<=2.5;y[mask]=1
    rows.append(dict(sha=sha,x=x,y=y,mask=mask,label=group['label'],times=ts,targets=targets))
    print(json.dumps({'features_ready':len(rows),'total':len(dev),'frames':len(ts),'seconds':round(time.monotonic()-started,2)}),flush=True)
del engine

def fit(rows,c):
    x=np.concatenate([r['x'][r['mask']] for r in rows]);y=np.concatenate([r['y'][r['mask']] for r in rows])
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=600,random_state=2026))
    model.fit(x,y);return model

reports=[];best=None
for c in [0.0001,0.001,0.01,0.1]:
    predictions=[]
    for i,held in enumerate(rows):
        model=fit([r for j,r in enumerate(rows) if j!=i],c)
        scores=model.predict_proba(held['x'])[:,1]
        predictions.append({'sha':held['sha'],'scores':scores.tolist(),'times':held['times'].tolist(),'label':held['label'],'targets':held['targets']})
    for threshold in [.25,.35,.45,.55,.65,.75,.85,.9,.95,.98]:
        tp=fp=tn=fn=localized=0
        for r in predictions:
            pred=max(r['scores'])>=threshold;y=r['label']
            tp+=bool(pred and y);fp+=bool(pred and not y);tn+=bool(not pred and not y);fn+=bool(not pred and y)
            if pred and y:
                t=r['times'][int(np.argmax(r['scores']))]
                localized+=min(abs(t-b) for b in r['targets'])<=2
        score=dict(C=c,threshold=threshold,tp=tp,fp=fp,tn=tn,fn=fn,accuracy=(tp+tn)/len(rows),f1=2*tp/max(1,2*tp+fp+fn),localization_recall_2s=localized/max(1,tp+fn))
        reports.append(score)
        rank=(score['f1'],score['accuracy'],score['localization_recall_2s'],-c)
        if best is None or rank>best[0]:best=(rank,score,predictions)
    print(json.dumps({'cv_done_C':c,'best':best[1]}),flush=True)
model=fit(rows,best[1]['C'])
(root/'models').mkdir(exist_ok=True)
joblib.dump({'pipeline':model,'threshold':best[1]['threshold'],'metadata':best[1],'feature_version':'qwen-mean-std-temporal-v1'},root/'models/event-detector.joblib')
(private/'detector-cv.json').write_text(json.dumps({'method':'leave-one-development-video-out; hyperparameters selected on these 24 groups, not independent test','best':best[1],'search':reports},indent=2))
(private/'detector-cv-predictions.json').write_text(json.dumps(best[2]))
print('DETECTOR_TRAINED',json.dumps(best[1]),flush=True)
