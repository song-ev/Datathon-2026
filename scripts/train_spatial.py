"""Region-aware visual features and grouped nonlinear development comparison."""
import json,os,time
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from forte.engine import Engine
from forte.spatial import extract_spatial,aggregate_regions
from scripts.train_lora import records,stats

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'));_,_,test,dev=records(root)
groups=sorted(dev+test,key=lambda p:p[0]);cache=root/'private/spatial-features';cache.mkdir(exist_ok=True)
paths={p.name:p for p in (root/'data').rglob('*.mov')};engine=None;xs={'motion':[],'context':[]};labels=[]
for sha,g in groups:
    file=cache/(sha+'.npz')
    if not file.exists():
        if engine is None:engine=Engine()
        times,vectors=extract_spatial(engine,paths[g['records'][0]['name']])
        np.savez_compressed(file,times=times,features=vectors.astype(np.float16))
    z=np.load(file)
    for mode in xs:xs[mode].append(aggregate_regions(z['features'],mode))
    labels.append(g['label']);print(json.dumps({'spatial_features_ready':len(labels),'total':len(groups)}),flush=True)
del engine;torch.cuda.empty_cache();torch.set_num_threads(2)
xs={k:np.stack(v) for k,v in xs.items()};labels=np.array(labels)
folds=list(StratifiedKFold(4,shuffle=True,random_state=2026).split(xs['motion'],labels))

def build(name):
    if name=='trees':return ExtraTreesClassifier(n_estimators=300,max_depth=4,max_features=.05,class_weight='balanced',random_state=2026,n_jobs=2)
    if name.startswith('rbf'):
        c=float(name.split('-')[1]);return make_pipeline(Normalizer(),SVC(C=c,kernel='rbf',gamma='scale',class_weight='balanced',probability=True,random_state=2026))
    return make_pipeline(StandardScaler(),LogisticRegression(C=.001,class_weight='balanced',solver='liblinear',random_state=2026,max_iter=1000))

history=[];best=None;predictions={}
for mode in xs:
    for name in ['linear','rbf-1','rbf-10','trees']:
        scores=np.zeros(len(labels))
        for train,valid in folds:
            m=build(name).fit(xs[mode][train],labels[train]);scores[valid]=m.predict_proba(xs[mode][valid])[:,1]
        predictions[mode+'/'+name]=scores.tolist()
        for threshold in [.3,.4,.5,.6,.7]:
            r=stats(labels,scores,threshold);r.update(mode=mode,estimator=name)
            eligible=r['accuracy']>=.6 and r['specificity']>=.5 and r['recall']>=.6
            rank=(eligible,r['f2'],r['accuracy'],r['specificity'])
            history.append(r)
            if best is None or rank>best[0]:best=(rank,r)
        print(json.dumps({'completed':mode+'/'+name,'selected':best[1],'eligible':best[0][0]}),flush=True)
r=best[1];model=build(r['estimator']).fit(xs[r['mode']],labels)
joblib.dump({'pipeline':model,'threshold':r['threshold'],'mode':r['mode'],'metadata':r,'training_groups':len(groups),'feature_version':'spatial-v1'},root/'models/spatial-detector.joblib')
(root/'private/spatial-cv-summary.json').write_text(json.dumps({'selected':r,'eligible':best[0][0],'search':history,'folds':4,'note':'OOF development model/threshold selection, not an independent test.'},indent=2))
(root/'private/spatial-cv-predictions.json').write_text(json.dumps({'sha256':[s for s,_ in groups],'labels':labels.tolist(),'scores':predictions}))
print(json.dumps({'final_fit_complete':True,'training_groups':len(groups),'selected':r,'eligible':best[0][0]}),flush=True)
