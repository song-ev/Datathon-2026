"""Post-evaluation compatibility check of scene validation; no classifier fitting."""
import json, os
from pathlib import Path
from forte.engine import Engine, probe, sample
from scripts.evaluate import split_records
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
e=Engine()
audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'))
paths={p.name:p for p in (root/'data').rglob('*.mov')}
results={}
for split,rows in zip(['dev','holdout'],split_records(audit['records'])):
    counts={'total':len(rows),'accepted':0,'rejected':0,'errors':0}
    for sha,g in rows:
        try:
            p=paths[g['records'][0]['name']];duration=probe(p)
            answer=e.infer(sample(p,step=max(1,duration/3))[:3],tokens=70,prompt='Decide whether these frames show a real indoor cashier/cash-handling workstation or bank service counter. Do not identify people. A real office counter with a worker is sufficient; cash need not be visible. Return ONLY JSON {"events":[{"cashier_scene":true}]} if yes. For cartoons, geometric shapes, blank images, outdoor scenes or unrelated content return {"events":[]}.')
            ok=any(x.get('cashier_scene') is True for x in answer['events'] if isinstance(x,dict))
            counts['accepted' if ok else 'rejected']+=1
        except Exception:
            counts['errors']+=1
    results[split]=counts
    print(json.dumps({split:counts}),flush=True)
(root/'private/scene-validation.json').write_text(json.dumps(results,indent=2))
