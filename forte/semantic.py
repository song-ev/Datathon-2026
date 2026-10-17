"""Local, label-free action features. Never exports video-derived observations."""
import math
import time
import numpy as np
from forte.engine import probe,sample,nearest

FEATURES=['cash_visible','client_visible','cashier_at_counter','manual_counting','machine_counting','cash_handover','document_handover','drawer_open','drawer_closed','cashier_leaves','cash_unattended','client_leaves','cash_to_personal_storage','phone_use','bag_transfer','multiple_cash_piles','cash_stored_in_drawer','cash_handover_without_visible_count']
PROMPT='''Analyze successive timestamped frames of a silent cashier video. Record visible actions, not legal conclusions. Do not identify people or infer intent. For each feature return 0=not visible, 1=uncertain, 2=clearly visible in at least one frame. An action requires temporal evidence; absence of evidence is not proof. Personal storage means pocket or personal bag, not cashier drawer. Unattended cash means cash left accessible while the cashier is away. Handover without visible count means a handover is visible but no count is seen within these frames; it is not by itself a violation. Return ONLY JSON {"events":[{"values":[18 integers in the exact feature order],"time_sec":timestamp_of_most_notable_action}]}. Feature order: '''+', '.join(FEATURES)

def extract_semantics(engine,path,progress=None,time_limit=230):
    start=time.monotonic();duration=probe(path);frames=sample(path)
    windows=[]
    # Full-context pass plus overlapping local passes; globals and locals have separate pooling.
    ranges=[(0,duration-.05,12)]
    ranges += [(a,min(duration-.05,a+16),8) for a in range(0,max(1,math.ceil(duration)-2),12)]
    for i,(begin,end,count) in enumerate(ranges):
        if time.monotonic()-start>time_limit-20:raise TimeoutError('Semantic feature budget')
        selected=[nearest(frames,begin+(end-begin)*j/(count-1)) for j in range(count)]
        answer=engine.infer(selected,prompt=PROMPT,tokens=230)
        values=answer['events'][0]['values'] if answer['events'] else []
        if len(values)!=len(FEATURES):
            answer=engine.infer(selected,prompt=PROMPT+' Exactly 18 integers are required.',tokens=230)
            values=answer['events'][0]['values']
        if len(values)!=len(FEATURES):raise ValueError('Invalid semantic feature length')
        v=np.array(values,dtype=float)
        if not np.isfinite(v).all() or np.any((v<0)|(v>2)):raise ValueError('Invalid semantic values')
        timestamp=float(answer['events'][0].get('time_sec',(begin+end)/2))
        if not math.isfinite(timestamp):timestamp=(begin+end)/2
        windows.append({'begin':begin,'end':end,'time_sec':float(nearest(frames,min(end,max(begin,timestamp)))[0]),'values':v.tolist()})
        if progress:progress(round((i+1)/len(ranges)*75),'Анализ действий и контекста')
    return windows

def aggregate(windows):
    g=np.array(windows[0]['values'])/2
    x=np.array([r['values'] for r in windows[1:]],dtype=float)/2
    return np.concatenate([g,x.mean(0),x.max(0),x.std(0)])
