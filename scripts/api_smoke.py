"""Run against a live API; print aggregates, never credentials or dataset content."""
import base64
import io
import json
import os
import time
from pathlib import Path

import av
import httpx
import numpy as np
from PIL import Image

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
key=(root/'private/api-key.txt').read_text().strip()
url=os.getenv('TEST_BASE_URL','http://127.0.0.1:8000')
c=httpx.Client(base_url=url,timeout=20)
h={'X-API-Key':key}
checks={}
checks['health']=c.get('/api/v1/health').json()['status']=='ready'
checks['unauthorized']=c.get('/api/v1/jobs/'+'0'*32).status_code==401
checks['wrong_key']=c.get('/api/v1/rules',headers={'X-API-Key':'wrong'}).status_code==401
checks['missing_job']=c.get('/api/v1/jobs/'+'0'*32,headers=h).status_code==404
checks['rules']=len(c.get('/api/v1/rules',headers=h).json()['rules'])>=7
checks['wrong_extension']=c.post('/api/v1/analyze',headers=h,files={'file':('x.txt',b'test')}).status_code==415
checks['invalid_mov']=c.post('/api/v1/analyze',headers=h,files={'file':('x.mov',b'not a movie')}).status_code==422
path=root/'private/synthetic.mov'
with av.open(str(path),'w') as out:
    stream=out.add_stream('libx264',rate=12);stream.width=384;stream.height=384;stream.pix_fmt='yuv420p'
    for i in range(36):
        image=np.full((384,384,3),245,dtype=np.uint8);image[130:190,20+i*5:80+i*5]=[200,30,60]
        frame=av.VideoFrame.from_ndarray(image,format='rgb24')
        for packet in stream.encode(frame):out.mux(packet)
    for packet in stream.encode():out.mux(packet)
with path.open('rb') as f:r=c.post('/api/v1/analyze',headers=h,files={'file':('synthetic.mov',f)})
checks['accepted']=r.status_code==202
start=time.monotonic()
if checks['accepted']:
    job_id=r.json()['job_id']
    while time.monotonic()-start<290:
        report=c.get('/api/v1/jobs/'+job_id,headers=h).json()
        if report['status']!='processing':break
        time.sleep(2)
    checks['synthetic_completed']=report['status']=='done'
    checks['synthetic_no_violation']=report.get('verdict')=='no_violation' and report.get('violations')==[]
checks['ui']=c.get('/').status_code==200
checks['security_headers']='nosniff'==c.get('/').headers.get('x-content-type-options')
result={'checks':checks,'all_passed':all(checks.values()),'seconds':round(time.monotonic()-start,2)}
(root/'private/api-smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['all_passed'] else 1)
