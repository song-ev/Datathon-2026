"""Scene-relative temporal features, computed only on the local GPU server."""
import numpy as np

def relative_features(vectors):
    x=np.asarray(vectors,dtype=np.float32)
    centered=x-np.median(x,axis=0,keepdims=True)
    result=[]
    for i in range(len(x)):
        lo=max(0,i-2);hi=min(len(x),i+3)
        result.append(np.concatenate([centered[lo:hi].mean(0),x[hi-1]-x[lo]]))
    return np.stack(result)

def frame_scores(pipeline,x):
    scores=pipeline.predict_proba(x)[:,1]
    return np.array([scores[max(0,i-1):min(len(scores),i+2)].mean() for i in range(len(scores))])
