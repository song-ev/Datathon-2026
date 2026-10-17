"""Calibrate saved adapters on six development videos; never reads comparison labels."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from peft import PeftModel
from forte.engine import MODEL_PATH
from scripts.train_lora import records,stats
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);args=p.parse_args()
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'));model_dir=Path(args.model_dir)
_,val,_,_=records(root);cache=root/'private/lora-window-inputs-v2'
torch.set_num_threads(4)
processor=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
ids=[processor.tokenizer.encode(str(i),add_special_tokens=False)[0] for i in (0,1)]
base=Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
model=None;best=None;history=[]
for epoch in range(1,5):
    path=model_dir/f'epoch-{epoch}'
    if not path.exists():continue
    name=f'epoch{epoch}'
    if model is None:model=PeftModel.from_pretrained(base,path,adapter_name=name,is_trainable=False)
    else:model.load_adapter(path,adapter_name=name,is_trainable=False)
    model.set_adapter(name);model.eval();scores=[];labels=[]
    with torch.no_grad():
        for sha,g in val:
            parts=[]
            for file in sorted(cache.glob(sha+'-eval-*.pt')):
                inputs={k:v.to('cuda') for k,v in torch.load(file,weights_only=True).items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs,logits_to_keep=1,use_cache=False).logits[0,-1,ids].float()
                parts.append(float(torch.softmax(logits,0)[1]))
            if not parts:raise ValueError('Missing validation windows')
            scores.append(max(parts));labels.append(g['label'])
    candidates=[]
    for threshold in [.3,.4,.5,.6,.65,.7,.8,.9,.95,.98,.995]:
        r=stats(labels,scores,threshold);r['epoch']=epoch;candidates.append(r)
        eligible=r['accuracy']>=2/3 and r['tp']>0 and r['tn']>0
        rank=(eligible,r['f2'],r['accuracy'],r['specificity'])
        if best is None or rank>best[0]:best=(rank,r)
    history+=candidates
    print(json.dumps({'calibrated_epoch':epoch,'best_recall_policy':best[1],'eligible':best[0][0]}),flush=True)
r=best[1]
(root/'private/lora-recall-calibration.json').write_text(json.dumps({'selected':r,'eligible':best[0][0],'history':history,'note':'Adaptive calibration on six development videos, not an independent test. Requires accuracy>=2/3 and at least one correctly detected example of each class.'},indent=2))
(model_dir/f"epoch-{r['epoch']}"/'decision.json').write_text(json.dumps({'threshold':r['threshold'],'epoch':r['epoch'],'local_windows':True,'policy':'recall','eligible':best[0][0]}))
