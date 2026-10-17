"""Server-only feature/serving parity check; print aggregates, never video content."""
import json,os
from pathlib import Path
import numpy as np,joblib
from forte.engine import Engine
from forte.spatial import aggregate_regions
from scripts.train_lora import records
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
_,_,test,dev=records(root);sha,g=next((s,g) for s,g in dev+test if g['label'])
path=next((root/'data').rglob(g['records'][0]['name']))
m=joblib.load(root/'models/fusion-detector.joblib')
x=np.load(root/'private/features'/(sha+'.npz'))['features'].astype(np.float32)
regions=np.load(root/'private/spatial-features'/(sha+'.npz'))['features']
aggregate=np.concatenate([x.mean(0),x.std(0),np.abs(np.diff(x,axis=0)).mean(0)])[None]
score=m['global_weight']*m['linear'].predict_proba(aggregate)[0,1]+(1-m['global_weight'])*m['spatial'].predict_proba(aggregate_regions(regions,'motion')[None])[0,1]
engine=Engine();r=engine.analyze(path,include_evidence=False)
result={'mode':engine.detector_mode,'training_groups':r['analysis']['classifier_training_groups'],
        'cached_score':float(score),'live_score':r['analysis']['detector_score'],
        'parity':bool(abs(score-r['analysis']['detector_score'])<1e-5),
        'seconds':r['analysis']['processing_sec'],'model':r['analysis']['model']}
(root/'private/fusion-serving-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
assert result['parity'] and result['training_groups']==40
