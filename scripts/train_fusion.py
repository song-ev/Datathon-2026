"""OOF-calibrated fusion of context and spatial motion; fit only open-data videos."""
import json,os
from pathlib import Path
import joblib,numpy as np
from scripts.train_lora import stats
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
a=json.loads((root/'private/open-cv-predictions.json').read_text())
b=json.loads((root/'private/spatial-cv-predictions.json').read_text())
assert a['sha256']==b['sha256'] and a['labels']==b['labels']
y=np.array(a['labels']);global_scores=np.array(a['linear']);spatial_scores=np.array(b['scores']['motion/linear'])
best=None;history=[]
for weight in [0,.25,.5,.75,1]:
    scores=weight*global_scores+(1-weight)*spatial_scores
    for threshold in np.arange(.25,.751,.025):
        m=stats(y,scores,float(round(threshold,3)));m['global_weight']=weight
        eligible=m['accuracy']>=.6 and m['specificity']>=.5 and m['recall']>=.6
        rank=(eligible,m['f2'],m['accuracy'],m['specificity'])
        history.append(m)
        if best is None or rank>best[0]:best=(rank,m)
selected=best[1]
spatial=joblib.load(root/'models/spatial-detector.joblib')
if spatial['mode']!='motion' or spatial['metadata']['estimator']!='linear':raise ValueError('Expected fixed motion/linear component')
linear=joblib.load(root/'models/open-detector.joblib')['linear']
joblib.dump({'spatial':spatial['pipeline'],'linear':linear,'threshold':selected['threshold'],'global_weight':selected['global_weight'],'training_groups':40,'metadata':selected},root/'models/fusion-detector.joblib')
report={'selected':selected,'eligible':best[0][0],'folds':4,'baseline':stats(y,global_scores,.5),'search':history,
        'note':'Adaptive OOF development selection of fusion weight and threshold, not an unbiased or hidden-test estimate.'}
(root/'private/fusion-cv-summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'selected':selected,'eligible':best[0][0],'baseline':report['baseline']}),flush=True)
