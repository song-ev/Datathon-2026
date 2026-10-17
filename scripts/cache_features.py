"""Cache frozen features for all open videos; never reads API uploads."""
import json,os
from pathlib import Path
import numpy as np
from forte.engine import Engine
from forte.features import extract
from scripts.train_lora import records
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
_,_,test,dev=records(root);paths={p.name:p for p in (root/'data').rglob('*.mov')}
cache=root/'private/features';cache.mkdir(exist_ok=True);engine=None
for i,(sha,g) in enumerate(dev+test):
    file=cache/(sha+'.npz')
    if not file.exists():
        if engine is None:engine=Engine()
        times,features=extract(engine,paths[g['records'][0]['name']])
        np.savez_compressed(file,times=times,features=features.astype(np.float16))
    print(json.dumps({'features_ready':i+1,'total':len(dev+test)}),flush=True)
