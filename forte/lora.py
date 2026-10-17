"""Shared input contract for LoRA training and serving."""
import random
from forte.engine import sample,nearest

PROMPT='These timestamped frames show one bank cashier operation. Classify observable procedural non-compliance. Consider cash handling, counting, handover, storage and whether the cashier leaves an unfinished transaction. Do not identify people or infer intentions. Reply with one digit only: 1 for violation, 0 for no violation.'

def inputs_for(processor,path,variant=0,bounds=None,frames=None):
    frames=sample(path) if frames is None else frames;count=12;rng=random.Random(2026+variant)
    last=frames[-1][0];content=[]
    begin,end=(0,last) if bounds is None else bounds
    for i in range(count):
        pos=i/(count-1)
        if variant:pos=min(1,max(0,pos+rng.uniform(-.035,.035)))
        t,im=nearest(frames,begin+(end-begin)*pos);im=im.copy();im.thumbnail((336,616) if bounds is not None else (192,288))
        content.extend([{'type':'text','text':f't={t-begin:.1f}s'},{'type':'image','image':im}])
    content.append({'type':'text','text':PROMPT})
    return processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
