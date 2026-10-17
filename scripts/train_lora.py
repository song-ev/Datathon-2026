"""Offline LoRA binary video classifier. Dataset and derived outputs never leave the GPU server."""
import argparse,hashlib,json,os,random,time,re,math
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
import numpy as np
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,PeftModel
from forte.engine import sample,nearest,probe,MODEL_PATH
from scripts.evaluate import split_records

from forte.lora import PROMPT,inputs_for

def records(root):
    audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'))
    dev,test=split_records(audit['records']);train=[];val=[]
    for label in (0,1):
        part=[r for r in dev if r[1]['label']==label]
        part.sort(key=lambda r:hashlib.sha256(('lora-val-v1'+r[0]).encode()).hexdigest())
        val+=part[:3];train+=part[3:]
    assert not {x[0] for x in train}&{x[0] for x in val}
    return train,val,test,dev

def stats(labels,scores,threshold=.5):
    y=np.array(labels);p=np.array(scores)>=threshold
    tp=int(np.sum(p&(y==1)));fp=int(np.sum(p&(y==0)));tn=int(np.sum(~p&(y==0)));fn=int(np.sum(~p&(y==1)))
    return dict(n=len(y),threshold=threshold,tp=tp,fp=fp,tn=tn,fn=fn,accuracy=(tp+tn)/len(y),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),specificity=tn/max(1,tn+fp),f1=2*tp/max(1,2*tp+fp+fn),f2=5*tp/max(1,5*tp+4*fn+fp))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial-steps',type=int,default=0);parser.add_argument('--epochs',type=int,default=4);parser.add_argument('--refit',action='store_true');parser.add_argument('--evaluate',type=str);parser.add_argument('--local-windows',action='store_true');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
    model_suffix='-8b' if '8B' in MODEL_PATH else ''
    prefix=('lora-local' if args.local_windows else 'lora')+model_suffix
    out=root/'models'/(prefix+('-refit' if args.refit else '-development'));out.mkdir(exist_ok=True)
    torch.manual_seed(2026);random.seed(2026);np.random.seed(2026);torch.set_num_threads(2 if args.prepare_only else 4)
    processor=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True,min_pixels=64*32*32,max_pixels=(192 if args.local_windows else 96)*32*32)
    train,val,test,dev=records(root);paths={p.name:p for p in (root/'data').rglob('*.mov')}
    cache=root/'private'/(('lora-window-inputs-v2' if args.local_windows else 'lora-inputs-v1')+model_suffix);cache.mkdir(exist_ok=True)
    validation_variants={}
    selected=(test if args.evaluate else (dev if args.refit else train+val))
    for sha,g in selected:
        is_validation=args.evaluate or (sha in {x[0] for x in val} and not args.refit)
        path=paths[g['records'][0]['name']]
        if args.local_windows:
            duration=probe(path);bounds=[];decoded=sample(path)
            if is_validation:
                bounds=[(float(a),min(duration-.05,a+12)) for a in range(0,max(1,math.ceil(duration)-2),6)]
                validation_variants[sha]=len(bounds)
            else:
                targets=[]
                for r in g['records']:
                    t=str(r.get('raw_time') or '');n=re.findall(r'\d+',t)
                    if n:targets.append(float(n[0])*60+float(n[1]) if 'мин' in t and len(n)>1 else float(n[0]))
                for v in range(3):
                    center=targets[v%len(targets)]+(v-1)*2 if g['label'] and targets else duration*(v+.5)/3
                    begin=max(0,min(duration-12,center-6));bounds.append((begin,min(duration-.05,begin+12)))
            for v,bound in enumerate(bounds):
                dest=cache/f"{sha}-{'eval' if is_validation else 'train'}-{v}.pt"
                if not dest.exists():torch.save(dict(inputs_for(processor,path,0,bound,frames=decoded)),dest)
        else:
            validation_variants[sha]=1
            for variant in ([0] if is_validation else [0,1,2]):
                dest=cache/f'{sha}-{variant}.pt'
                if not dest.exists():torch.save(dict(inputs_for(processor,path,variant)),dest)
    print(json.dumps({'stage':'inputs_ready','training_groups':len(dev if args.refit else train),'validation_groups':len(val),'test_groups':len(test)}),flush=True)
    if args.prepare_only:return
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    if args.evaluate:
        model=PeftModel.from_pretrained(model,args.evaluate,is_trainable=False)
    else:
        model.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=r'.*language_model.*\.(q_proj|v_proj|k_proj|o_proj)$',lora_dropout=.05,bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    ids=[processor.tokenizer.encode(str(i),add_special_tokens=False) for i in (0,1)]
    assert all(len(x)==1 for x in ids);ids=[x[0] for x in ids]
    def load(sha,v=0,evaluation=False):
        filename=f"{sha}-{'eval' if evaluation else 'train'}-{v}.pt" if args.local_windows else f'{sha}-{v}.pt'
        return {k:t.to('cuda') for k,t in torch.load(cache/filename,weights_only=True).items()}
    def evaluate(rows):
        model.eval();scores=[];labels=[]
        with torch.no_grad():
            for sha,g in rows:
                per_window=[]
                for variant in range(validation_variants[sha]):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(**load(sha,variant,evaluation=True),logits_to_keep=1,use_cache=False).logits[0,-1,ids].float()
                    per_window.append(float(torch.softmax(logits,0)[1]))
                scores.append(max(per_window));labels.append(g['label'])
        return labels,scores
    if args.evaluate:
        y,s=evaluate(test);threshold=json.loads((Path(args.evaluate)/'decision.json').read_text()).get('threshold',.5) if (Path(args.evaluate)/'decision.json').exists() else .5;report=stats(y,s,threshold)
        (root/'private'/(prefix+'-comparison.json')).write_text(json.dumps(report,indent=2))
        (root/'private'/(prefix+'-comparison-predictions.json')).write_text(json.dumps({'labels':y,'scores':s}))
        print(json.dumps({'comparison_previously_used_holdout':report}),flush=True);return
    params=[p for p in model.parameters() if p.requires_grad]
    print(json.dumps({'stage':'model_ready','trainable_parameters':sum(p.numel() for p in params),'allocated_gib':torch.cuda.memory_allocated()/2**30}),flush=True)
    if not args.trial_steps and not args.refit:
        base_y,base_scores=evaluate(val)
        print(json.dumps({'validation_before_training':stats(base_y,base_scores)}),flush=True)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    best=None;history=[];step=0;started=time.monotonic();training=dev if args.refit else train
    for epoch in range(1,args.epochs+1):
        examples=[(sha,g['label'],v) for sha,g in training for v in (0,1,2)];random.Random(2026+epoch).shuffle(examples)
        model.train();optimizer.zero_grad();loss_sum=0
        for i,(sha,label,variant) in enumerate(examples):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(**load(sha,variant),logits_to_keep=1,use_cache=False).logits[0,-1,ids].float()
                loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([label],device='cuda'))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            (loss/4).backward();loss_sum+=float(loss.detach());step+=1
            if (i+1)%4==0 or i+1==len(examples):
                torch.nn.utils.clip_grad_norm_(params,1);optimizer.step();optimizer.zero_grad()
            if step%4==0:
                print(json.dumps({'epoch':epoch,'step':step,'mean_loss':round(loss_sum/(i+1),4),'elapsed_sec':round(time.monotonic()-started,1),'peak_gib':round(torch.cuda.max_memory_allocated()/2**30,2)}),flush=True)
            if args.trial_steps and step>=args.trial_steps:
                print(json.dumps({'trial_completed':True,'steps':step,'seconds':time.monotonic()-started,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'adapter_b_l1':sum(float(p.detach().abs().sum()) for n,p in model.named_parameters() if 'lora_B' in n)}),flush=True);return
        model.save_pretrained(out/f'epoch-{epoch}')
        if not args.refit:
            y,s=evaluate(val)
            candidates=[stats(y,s,t) for t in ([.5,.65,.8,.9,.95,.98,.995] if args.local_windows else [.5])]
            report=max(candidates,key=lambda r:(r['f2'] if r['specificity']>=2/3 else -1,r['accuracy']))
            rank=(report['f2'] if report['specificity']>=2/3 else -1,report['accuracy'])
            history.append({'epoch':epoch,'metrics':report})
            if best is None or rank>best[0]:
                best=(rank,epoch);model.save_pretrained(out/'best');(out/'best/decision.json').write_text(json.dumps({'threshold':report['threshold'],'epoch':epoch,'local_windows':args.local_windows}))
            (root/'private'/(prefix+'-development.json')).write_text(json.dumps({'history':history,'best_epoch':best[1],'note':'Six development validation groups; previously used holdout not inspected during fitting.'},indent=2))
            print(json.dumps({'epoch':epoch,'validation':report,'best_epoch':best[1]}),flush=True)
        else:print(json.dumps({'refit_epoch_complete':epoch}),flush=True)
    print(json.dumps({'training_complete':True,'elapsed_sec':round(time.monotonic()-started,1)}),flush=True)

if __name__=='__main__':main()
