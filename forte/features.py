"""Trainable detector on local frozen Qwen visual embeddings."""
import numpy as np
import torch
from forte.engine import sample

def extract(engine,path,step=1.0):
    frames=sample(path,step=step)
    vectors=[]
    with torch.inference_mode():
        for begin in range(0,len(frames),8):
            batch=frames[begin:begin+8]
            inputs=engine.processor.image_processor(images=[im for _,im in batch],return_tensors='pt').to('cuda')
            embeddings,_=engine.model.model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
            for emb in embeddings:
                emb=emb.float()
                vectors.append(torch.cat([emb.mean(0),emb.std(0)]).cpu().numpy())
            del inputs,embeddings
    return np.array([t for t,_ in frames]),np.stack(vectors)

def temporal(vectors):
    result=[]
    for i in range(len(vectors)):
        lo=max(0,i-2);hi=min(len(vectors),i+3)
        window=vectors[lo:hi]
        result.append(np.concatenate([window.mean(0),np.abs(window[-1]-window[0])]))
    return np.stack(result).astype(np.float32)

def peaks(times,scores,threshold):
    selected=[]
    for i in np.argsort(-scores):
        if scores[i]<threshold:break
        if not any(abs(times[i]-x['time_sec'])<6 for x in selected):
            selected.append({'time_sec':float(times[i]),'confidence':float(scores[i])})
        if len(selected)>=3:break
    return sorted(selected,key=lambda x:x['time_sec'])
