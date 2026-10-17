"""Fit a frame localizer using only supplied open-data timestamps; evaluate by video."""
import json,os,re
from pathlib import Path
import joblib,numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from forte.spatial import frame_regions
from scripts.train_lora import records
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'));_,_,test,dev=records(root);groups=sorted(dev+test,key=lambda p:p[0]);rows=[]
for sha,g in groups:
    z=np.load(root/'private/spatial-features'/(sha+'.npz'));targets=[]
    for r in g['records']:
        raw=str(r.get('raw_time') or '');n=re.findall(r'\d+',raw)
        if n:targets.append(float(n[0])*60+float(n[1]) if 'мин' in raw and len(n)>1 else float(n[0]))
    mask=np.abs(z['times'][:,None]-np.array(targets)[None,:]).min(1)<=2.5 if g['label'] and targets else np.ones(len(z['times']),dtype=bool)
    rows.append({'x':frame_regions(z['features']),'mask':mask,'label':g['label'],'times':z['times'],'targets':targets})
def fit(indices):
    selected=[rows[i] for i in indices];x=np.concatenate([r['x'][r['mask']] for r in selected]);y=np.concatenate([np.full(r['mask'].sum(),r['label']) for r in selected])
    weights=np.concatenate([np.full(r['mask'].sum(),1/max(1,r['mask'].sum())) for r in selected]);weights*=len(weights)/weights.sum()
    return make_pipeline(Normalizer(),LogisticRegression(C=1,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=2026)).fit(x,y,logisticregression__sample_weight=weights)
labels=[r['label'] for r in rows];localized=0
for fold,(train,valid) in enumerate(StratifiedKFold(4,shuffle=True,random_state=2026).split(np.arange(len(rows)),labels)):
    model=fit(train)
    for i in valid:
        r=rows[i]
        if not r['label']:continue
        scores=model.predict_proba(r['x'])[:,1];t=float(r['times'][int(scores.argmax())])
        localized+=bool(r['targets'] and min(abs(t-b) for b in r['targets'])<=2)
    print(json.dumps({'localizer_fold':fold+1}),flush=True)
model=fit(range(len(rows)));joblib.dump({'pipeline':model,'training_groups':len(rows)},root/'models/spatial-localizer.joblib')
report={'positive_videos':sum(labels),'localized_within_2s':localized,'localization_recall_2s':localized/sum(labels),'folds':4,'training_groups_final':len(rows)}
(root/'private/spatial-localization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
