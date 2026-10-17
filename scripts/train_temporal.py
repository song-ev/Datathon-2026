"""Temporal multiple-instance learning; training-only PCA, video-group validation."""
import argparse, copy, json, os, re, random
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from forte.temporal_model import TemporalDetector, transform, clip_logit
from scripts.train_lora import records, stats

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
    train,val,test,_=records(root)
    torch.manual_seed(2026);np.random.seed(2026);random.seed(2026);torch.set_num_threads(2)
    artifact=root/'models/temporal-detector.joblib'
    rows={}
    engine=None
    paths={p.name:p for p in (root/'data').rglob('*.mov')}
    for sha,g in (test if args.evaluate else train+val):
        file=root/'private/features'/(sha+'.npz')
        if not file.exists():
            from forte.engine import Engine
            from forte.features import extract
            if engine is None:engine=Engine()
            times,features=extract(engine,paths[g['records'][0]['name']])
            np.savez_compressed(file,times=times,features=features.astype(np.float16))
            print(json.dumps({'features_ready':len(rows)+1}),flush=True)
        z=np.load(file)
        targets=[]
        for r in g['records']:
            raw=str(r.get('raw_time') or '');n=re.findall(r'\d+',raw)
            if n:targets.append(float(n[0])*60+float(n[1]) if 'мин' in raw and len(n)>1 else float(n[0]))
        rows[sha]={'raw':z['features'],'times':z['times'],'targets':targets,'label':g['label']}
    if args.evaluate:
        saved=joblib.load(artifact);pca=saved['pca']
    else:
        # Equal contribution per training video; no PCA fitted on validation/test.
        xs=[]
        for sha,_ in train:
            x=rows[sha]['raw'];x=x[np.linspace(0,len(x)-1,min(24,len(x))).astype(int)]
            xs.append(x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8))
        pca=PCA(n_components=32,whiten=True,random_state=2026).fit(np.concatenate(xs))
    for r in rows.values():
        r['x']=torch.from_numpy(transform(r['raw'],pca))
        if r['label'] and r['targets']:
            d=np.abs(r['times'][:,None]-np.array(r['targets'])[None,:]).min(1)
            r['mask']=torch.from_numpy(d<=2.5)
        else:r['mask']=torch.ones(len(r['x']),dtype=torch.bool)
    model=TemporalDetector()
    def evaluate(part):
        model.eval();labels=[];scores=[]
        with torch.no_grad():
            for sha,g in part:
                scores.append(float(torch.sigmoid(clip_logit(model(rows[sha]['x'])))))
                labels.append(g['label'])
        return labels,scores
    if args.evaluate:
        model.load_state_dict(saved['state']);y,s=evaluate(test)
        result=stats(y,s,saved['threshold'])
        (root/'private/temporal-comparison.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'comparison_previously_used_holdout':result}));return
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
    best=None;history=[]
    for epoch in range(1,101):
        model.train();order=list(train);random.shuffle(order);total=0
        for sha,g in order:
            r=rows[sha];x=r['x']+torch.randn_like(r['x'])*.04
            logits=model(x);target=torch.tensor(float(g['label']))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(clip_logit(logits),target)
            if r['mask'].any():
                loss=loss+.5*torch.nn.functional.binary_cross_entropy_with_logits(logits[r['mask']],target.expand(int(r['mask'].sum())))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())
        if epoch%10:continue
        y,s=evaluate(val)
        for threshold in [.3,.4,.5,.6,.7,.8,.9]:
            m=stats(y,s,threshold);m['epoch']=epoch
            eligible=m['accuracy']>=2/3 and m['tp']>0 and m['tn']>0
            rank=(eligible,m['f2'],m['accuracy'],m['specificity'])
            history.append(m)
            if best is None or rank>best[0]:best=(rank,m,copy.deepcopy(model.state_dict()))
        print(json.dumps({'epoch':epoch,'loss':round(total/len(train),4),'best_validation':best[1],'eligible':best[0][0]}),flush=True)
    joblib.dump({'pca':pca,'state':best[2],'threshold':best[1]['threshold'],'metadata':best[1],'training_groups':18,'feature_version':'temporal-mil-v1'},artifact)
    (root/'private/temporal-validation.json').write_text(json.dumps({'selected':best[1],'eligible':best[0][0],'history':history,'note':'Adaptive development selection on six videos; not an independent performance estimate.'},indent=2))

if __name__=='__main__':main()
