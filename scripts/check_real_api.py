"""Validate evidence-bearing response with one already evaluated real video, on server only."""
import base64, io, json, os, time
from pathlib import Path
import httpx
from PIL import Image
root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
rows=json.loads((root/'private/evaluation-holdout.json').read_text())
sha=next(r['sha256'] for r in rows if r['label'] and r.get('candidates'))
audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'))
name=next(r['name'] for r in audit['records'] if r['sha256']==sha)
p=next((root/'data').rglob(name))
c=httpx.Client(base_url='http://127.0.0.1:8000',timeout=20,headers={'X-API-Key':(root/'private/api-key.txt').read_text().strip()})
with p.open('rb') as f:r=c.post('/api/v1/analyze',files={'file':(name,f)})
r.raise_for_status();jid=r.json()['job_id'];start=time.monotonic()
while time.monotonic()-start<290:
    j=c.get('/api/v1/jobs/'+jid).json()
    if j['status']!='processing':break
    time.sleep(2)
checks={'done':j['status']=='done','violation':j.get('verdict')=='violation','events':len(j.get('violations',[]))}
images=[]
for v in j.get('violations',[]):
    for f in v['evidence_frames']:
        im=Image.open(io.BytesIO(base64.b64decode(f['image_base64'])))
        images.append(im.format=='JPEG' and im.size==(464,848) and abs(f['timestamp_sec']-v['timecode_sec'])<=1.1)
checks['evidence_jpegs']=len(images);checks['original_size_and_time_valid']=bool(images) and all(images)
checks['model']=j.get('analysis',{}).get('model');checks['seconds']=j.get('analysis',{}).get('processing_sec')
(root/'private/real-api-check.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks),flush=True)

