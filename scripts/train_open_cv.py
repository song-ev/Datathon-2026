"""Four-fold grouped development evaluation, then fit on every open-data video.

Does not access API uploads or hidden jury tests. Hyperparameters were explored on
this same open dataset: these are development estimates, not an independent test.
"""
import copy,json,os,random,re
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from forte.temporal_model import TemporalDetector,transform,clip_logit
from scripts.train_lora import records,stats

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
_,_,test,dev=records(root);groups=sorted(dev+test,key=lambda p:p[0]);rows=[]
torch.set_num_threads(2)
for sha,g in groups:
    z=np.load(root/'private/features'/(sha+'.npz'));x=z['features'].astype(np.float32);t=z['times']
    targets=[]
    for r in g['records']:
        raw=str(r.get('raw_time') or '');n=re.findall(r'\d+',raw)
        if n:targets.append(float(n[0])*60+float(n[1]) if 'мин' in raw and len(n)>1 else float(n[0]))
    mask=(np.abs(t[:,None]-np.array(targets)[None,:]).min(1)<=2.5) if g['label'] and targets else np.ones(len(t),dtype=bool)
    aggregate=np.concatenate([x.mean(0),x.std(0),np.abs(np.diff(x,axis=0)).mean(0) if len(x)>1 else np.zeros(x.shape[1])])
    rows.append({'x':x,'mask':torch.from_numpy(mask),'label':g['label'],'aggregate':aggregate,'times':t,'targets':targets})
labels=np.array([r['label'] for r in rows]);aggregate=np.stack([r['aggregate'] for r in rows])

def fit_temporal(indices):
    torch.manual_seed(2026);np.random.seed(2026);rng=random.Random(2026)
    xs=[]
    for i in indices:
        x=rows[i]['x'];x=x[np.linspace(0,len(x)-1,min(24,len(x))).astype(int)]
        xs.append(x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8))
    pca=PCA(n_components=32,whiten=True,random_state=2026).fit(np.concatenate(xs))
    inputs={i:torch.from_numpy(transform(rows[i]['x'],pca)) for i in indices}
    model=TemporalDetector();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
    for epoch in range(10):
        order=list(indices);rng.shuffle(order);model.train()
        for i in order:
            r=rows[i];x=inputs[i];logits=model(x+torch.randn_like(x)*.04);target=torch.tensor(float(r['label']))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(clip_logit(logits),target)
            if r['mask'].any():loss=loss+.5*torch.nn.functional.binary_cross_entropy_with_logits(logits[r['mask']],target.expand(int(r['mask'].sum())))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    return pca,model.eval()

oof=np.zeros(len(rows));baseline=np.zeros(len(rows));localization=np.zeros(len(rows),dtype=bool);fold_ids=np.zeros(len(rows),dtype=int)
for fold,(train,valid) in enumerate(StratifiedKFold(4,shuffle=True,random_state=2026).split(aggregate,labels)):
    assert not set(train)&set(valid)
    linear=make_pipeline(StandardScaler(),LogisticRegression(C=.0001,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=2026)).fit(aggregate[train],labels[train])
    baseline[valid]=linear.predict_proba(aggregate[valid])[:,1]
    pca,model=fit_temporal(train)
    with torch.no_grad():
        for i in valid:
            r=rows[i];logits=model(torch.from_numpy(transform(r['x'],pca)))
            oof[i]=float(torch.sigmoid(clip_logit(logits)));fold_ids[i]=fold
            if r['targets']:localization[i]=min(abs(float(r['times'][int(logits.argmax())])-t) for t in r['targets'])<=2
    print(json.dumps({'fold_complete':fold+1,'train_groups':len(train),'validation_groups':len(valid)}),flush=True)

# Select deployment operating point using out-of-fold predictions only.
history=[];best=None
for mode,scores in [('temporal',oof),('ensemble',(oof+baseline)/2),('linear',baseline)]:
    for threshold in [.3,.4,.5,.6,.7]:
        m=stats(labels,scores,threshold);m['mode']=mode
        eligible=m['accuracy']>=.6 and m['specificity']>=.5
        rank=(eligible,m['f2'],m['accuracy'],m['specificity'])
        history.append(m)
        if best is None or rank>best[0]:best=(rank,m)
baseline_metrics=stats(labels,baseline,.5)
report={'n_unique_open_videos':len(rows),'folds':4,'baseline_fixed':baseline_metrics,'selected':best[1],'eligible':best[0][0],
        'search':history,'note':'Out-of-fold development estimate. Operating point and architecture selected on these predictions; not an independent jury-test estimate.',
        'temporal_localization_recall_2s':float(np.sum(localization&(labels==1))/sum(labels==1))}
(root/'private/open-cv-summary.json').write_text(json.dumps(report,indent=2))
(root/'private/open-cv-predictions.json').write_text(json.dumps({'sha256':[s for s,_ in groups],'labels':labels.tolist(),'temporal':oof.tolist(),'linear':baseline.tolist(),'folds':fold_ids.tolist()}))
print(json.dumps({'baseline_fixed':baseline_metrics,'selected':best[1],'temporal_localization_recall_2s':report['temporal_localization_recall_2s']}),flush=True)
pca,model=fit_temporal(np.arange(len(rows)))
linear=make_pipeline(StandardScaler(),LogisticRegression(C=.0001,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=2026)).fit(aggregate,labels)
joblib.dump({'pca':pca,'state':model.state_dict(),'linear':linear,'threshold':best[1]['threshold'],'mode':best[1]['mode'],'metadata':best[1],
             'training_groups':len(rows),'feature_version':'temporal-mil-open-v1'},root/'models/open-detector.joblib')
print(json.dumps({'final_fit_complete':True,'training_groups':len(rows),'selected_mode':best[1]['mode']}),flush=True)
