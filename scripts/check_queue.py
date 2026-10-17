"""Server-only burst test; output timings and statuses, never video content."""
import json, os, time
from pathlib import Path
import av
import httpx

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'))
seen=set();videos=[]
for row in audit['records']:
    if row['sha256'] in seen:continue
    seen.add(row['sha256'])
    path=next((root/'data').rglob(row['name']))
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        duration=float(stream.duration*stream.time_base) if stream.duration else float(container.duration)/av.time_base
    videos.append((duration,path))
videos=sorted(videos,reverse=True,key=lambda x:x[0])[:5]
client=httpx.Client(base_url=os.getenv('TEST_BASE_URL','http://127.0.0.1:8000'),timeout=30,
                   headers={'X-API-Key':(root/'private/api-key.txt').read_text().strip()})
submitted=[];codes=[];max_queue=0
# The synthetic scene must fail without preventing the five real jobs behind it.
synthetic=root/'private/synthetic.mov'
sequence=([(0,synthetic)] if synthetic.exists() else [])+videos
for duration,path in sequence:
    started=time.monotonic()
    with path.open('rb') as file:
        response=client.post('/api/v1/analyze',files={'file':('test.mov',file,'video/quicktime')})
    codes.append(response.status_code);response.raise_for_status()
    submitted.append({'id':response.json()['job_id'],'start':started,'duration':duration,'synthetic':path==synthetic})
    max_queue=max(max_queue,client.get('/api/v1/health').json().get('queue_length',0))
invalid=client.post('/api/v1/analyze',files={'file':('invalid.mov',b'not a movie','video/quicktime')})
results=[]
while submitted and time.monotonic()-min(x['start'] for x in submitted)<310:
    for item in list(submitted):
        response=client.get('/api/v1/jobs/'+item['id']);response.raise_for_status();job=response.json()
        if job['status']=='processing':continue
        results.append({'synthetic':item['synthetic'],'video_seconds':round(item['duration'],3),
                        'status':job['status'],'error_code':job.get('error',{}).get('code'),
                        'total_seconds':round(time.monotonic()-item['start'],3),
                        'queue_seconds':round(job.get('started_at',job['created_at'])-job['created_at'],3)})
        submitted.remove(item)
    if submitted:time.sleep(0.5)
real=[r for r in results if not r['synthetic']]
report={'post_codes':codes,'invalid_upload_code':invalid.status_code,'max_queue_length':max_queue,
        'results':results,'unfinished':len(submitted),
        'passed':len(real)==5 and all(r['status']=='done' and r['total_seconds']<300 for r in real)
                 and all(c==202 for c in codes) and invalid.status_code==422 and max_queue>0
                 and all(r['status']=='failed' and r['error_code']=='UnsupportedScene' for r in results if r['synthetic'])}
(root/'private/queue-check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
raise SystemExit(0 if report['passed'] else 1)
