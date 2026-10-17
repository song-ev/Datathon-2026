import argparse,json,os,time
from pathlib import Path
from forte.engine import Engine
from forte.semantic import extract_semantics
from scripts.evaluate import split_records
p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','holdout'],default='dev');args=p.parse_args()
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'));cache=root/'private/semantic-v1';cache.mkdir(exist_ok=True)
audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'))
dev,hold=split_records(audit['records']);rows=dev if args.split=='dev' else hold
paths={p.name:p for p in (root/'data').rglob('*.mov')};e=Engine()
for i,(sha,g) in enumerate(rows):
    dst=cache/(sha+'.json');start=time.monotonic()
    if not dst.exists():
        try:
            result=extract_semantics(e,paths[g['records'][0]['name']])
            dst.write_text(json.dumps(result))
        except Exception as exc:
            print(json.dumps({'completed':i+1,'error':type(exc).__name__}),flush=True);continue
    print(json.dumps({'split':args.split,'completed':i+1,'total':len(rows),'seconds':round(time.monotonic()-start,2)}),flush=True)
