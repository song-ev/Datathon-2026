"""Small training-only search; no fitting on the comparison set."""
import json,os,re
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scripts.train_lora import records,stats
from forte.relative import relative_features,frame_scores
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'));train,val,_,_=records(root)
rows={}
for sha,g in train+val:
    z=np.load(root/'private/features'/(sha+'.npz'));t=z['times'];x=relative_features(z['features'])
    targets=[]
    for r in g['records']:
        s=str(r.get('raw_time') or '');n=re.findall(r'\d+',s)
        if n:targets.append(float(n[0])*60+float(n[1]) if 'мин' in s and len(n)>1 else float(n[0]))
    if g['label']:
        distance=np.min(np.abs(t[:,None]-np.array(targets)[None,:]),axis=1)
        mask=distance<=2.5;y=(distance<=2.5).astype(int)
    else:mask=np.ones(len(t),dtype=bool);y=np.zeros(len(t),dtype=int)
    rows[sha]=dict(x=x,t=t,y=y,mask=mask,label=g['label'],targets=targets)
train_rows=[rows[s] for s,_ in train]
x=np.concatenate([r['x'][r['mask']] for r in train_rows]);y=np.concatenate([r['y'][r['mask']] for r in train_rows])
weights=np.concatenate([np.full(r['mask'].sum(),1/r['mask'].sum()) for r in train_rows]);weights*=len(weights)/weights.sum()
best=None;history=[]
for c in [.1,1,10]:
    m=make_pipeline(Normalizer(),LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=800,random_state=2026))
    m.fit(x,y,logisticregression__sample_weight=weights)
    scores=[];labels=[];times=[]
    for sha,g in val:
        r=rows[sha];s=frame_scores(m,r['x']);scores.append(float(s.max()));labels.append(g['label']);times.append(float(r['t'][int(s.argmax())]))
    for threshold in [.3,.4,.5,.6,.7,.8,.9]:
        r=stats(labels,scores,threshold);r['C']=c
        rank=(r['f2'] if r['specificity']>=2/3 else -1,r['accuracy'])
        history.append(r)
        if best is None or rank>best[0]:best=(rank,m,r)
    print(json.dumps({'C':c,'best':best[2]}),flush=True)
joblib.dump({'pipeline':best[1],'threshold':best[2]['threshold'],'metadata':best[2],'training_groups':18,'feature_version':'relative-temporal-v1'},root/'models/relative-detector.joblib')
(root/'private/relative-validation.json').write_text(json.dumps({'best':best[2],'history':history},indent=2))

