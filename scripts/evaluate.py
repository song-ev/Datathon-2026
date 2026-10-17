"""Keep video-derived text on server. Print only aggregate metrics."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from forte.engine import Engine

def split_records(records):
    groups = {}
    for r in records:
        g = groups.setdefault(r['sha256'], {'records':[], 'label':int(r['label']=='нарушение')})
        g['records'].append(r)
    dev, holdout = [], []
    for label in (0,1):
        rows = [(k,g) for k,g in groups.items() if g['label']==label]
        rows.sort(key=lambda x:hashlib.sha256(('forte-2026-v1-'+x[0]).encode()).hexdigest())
        for i,(sha,g) in enumerate(rows):
            (holdout if i<8 else dev).append((sha,g))
    return dev,holdout

def metrics(rows, threshold):
    tp=fp=tn=fn=0
    errors=[]
    for r in rows:
        pred=any(x['confidence']>=threshold for x in r.get('candidates',[]))
        y=r['label']
        tp+=int(pred and y); fp+=int(pred and not y); tn+=int(not pred and not y); fn+=int(not pred and y)
        if y and pred and r['times']:
            pt=[x['time_sec'] for x in r['candidates'] if x['confidence']>=threshold]
            errors.append(min(abs(a-b) for a in pt for b in r['times']))
    return dict(n=len(rows),threshold=threshold,tp=tp,fp=fp,tn=tn,fn=fn,
        accuracy=round((tp+tn)/max(1,len(rows)),4),precision=round(tp/max(1,tp+fp),4),
        recall=round(tp/max(1,tp+fn),4),f1=round(2*tp/max(1,2*tp+fp+fn),4),
        detected_positives_within_2s=sum(x<=2 for x in errors),
        positive_localization_recall_2s=round(sum(x<=2 for x in errors)/max(1,tp+fn),4),
        processed=sum(not r.get('error') for r in rows),
        max_processing_sec=max((r.get('processing_sec',0) for r in rows),default=0))

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','holdout'],default='dev');p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
    audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'))
    dev,holdout=split_records(audit['records']); selected=dev if args.split=='dev' else holdout
    if args.limit: selected=selected[:args.limit]
    out=root/'private';out.mkdir(exist_ok=True)
    result_file=out/f'evaluation-{args.split}.json'
    done=json.loads(result_file.read_text()) if result_file.exists() else []
    engine=Engine()
    pending=[(sha,g) for sha,g in selected if not any(r['sha256']==sha for r in done)]
    wait_started=time.monotonic()
    while pending:
        paths={p.name:p for p in (root/'data').rglob('*.mov')}
        available=[(sha,g) for sha,g in pending if g['records'][0]['name'] in paths]
        if not available:
            if time.monotonic()-wait_started>7200:raise TimeoutError('Dataset upload timeout')
            time.sleep(10);continue
        sha,g=available[0];pending.remove((sha,g))
        rec=g['records'][0]
        times=[]
        for r in g['records']:
            s=r.get('raw_time') or ''
            numbers=re.findall(r'\d+',str(s))
            if numbers: times.append(float(numbers[0])*60+float(numbers[1]) if 'мин' in s and len(numbers)>1 else float(numbers[0]))
        row={'sha256':sha,'label':g['label'],'times':times}
        try:
            prediction=engine.analyze(paths[rec['name']],include_evidence=False)
            row.update(candidates=prediction.pop('_candidates'),processing_sec=prediction['analysis']['processing_sec'])
        except Exception as e:
            row.update(error=type(e).__name__,candidates=[])
        done.append(row);result_file.write_text(json.dumps(done,ensure_ascii=False,indent=2))
        print(json.dumps({'split':args.split,'completed':len(done),'planned':len(selected),'last_seconds':row.get('processing_sec'),'error':row.get('error')}),flush=True)
    thresholds=[0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9] if args.split=='dev' else [engine.threshold]
    report={'split':args.split,'metrics':[metrics(done,t) for t in thresholds],
        'notes':'SHA256 duplicate groups; 24 development / 16 holdout. Scenario similarity is not annotated. Types cannot be evaluated: labels contain only binary verdict and timestamp.'}
    (out/f'metrics-{args.split}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
