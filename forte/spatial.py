"""Preserve spatial regions instead of pooling the entire visual token grid."""
import numpy as np
import torch
from forte.engine import sample

def pool_regions(emb,grid,merge=2):
    height,width=int(grid[-2])//merge,int(grid[-1])//merge
    if height*width != len(emb):raise ValueError('Unexpected visual grid')
    a=emb.float().reshape(height,width,-1)
    h,w=max(1,height//2),max(1,width//2)
    regions=[a[:h,:w],a[:h,w:],a[h:,:w],a[h:,w:],a[height//4:max(height//4+1,3*height//4),width//4:max(width//4+1,3*width//4)]]
    return torch.stack([r.reshape(-1,a.shape[-1]).mean(0) for r in regions])

def extract_spatial(engine,path,include_global=False):
    frames=sample(path,step=1);vectors=[];global_vectors=[]
    with torch.inference_mode():
        for begin in range(0,len(frames),8):
            batch=frames[begin:begin+8]
            inputs=engine.processor.image_processor(images=[im for _,im in batch],return_tensors='pt').to('cuda')
            embeddings,_=engine.model.model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
            merge=engine.model.config.vision_config.spatial_merge_size
            for emb,grid in zip(embeddings,inputs['image_grid_thw']):
                vectors.append(pool_regions(emb,grid,merge).cpu().numpy())
                if include_global:
                    f=emb.float();global_vectors.append(torch.cat([f.mean(0),f.std(0)]).cpu().numpy())
    result=(np.array([t for t,_ in frames]),np.stack(vectors))
    return result+(np.stack(global_vectors),) if include_global else result

def aggregate_regions(x,mode):
    x=x.astype(np.float32)
    delta=np.abs(np.diff(x,axis=0)) if len(x)>1 else np.zeros_like(x)
    motion=np.concatenate([x.std(0).ravel(),delta.mean(0).ravel(),np.quantile(delta,.9,axis=0).ravel()])
    return motion if mode=='motion' else np.concatenate([x.mean(0).ravel(),motion])

def frame_regions(x):
    x=x.astype(np.float32).reshape(len(x),-1)
    before=np.stack([x[max(0,i-2):i+1].mean(0) for i in range(len(x))])
    after=np.stack([x[i:min(len(x),i+3)].mean(0) for i in range(len(x))])
    return np.concatenate([(before+after)/2,after-before],axis=1)
