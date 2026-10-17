"""Authenticated FastAPI service with a single supervised local GPU worker."""
import asyncio
import hmac
import json
import multiprocessing as mp
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(os.getenv('FORTE_HOME', '/home/ubuntu/forte-datathon'))
RUNTIME = ROOT/'runtime'
RUNTIME.mkdir(parents=True, exist_ok=True)
KEY_FILE = ROOT/'private'/'api-key.txt'
KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
if not KEY_FILE.exists():
    KEY_FILE.write_text(secrets.token_urlsafe(32))
    KEY_FILE.chmod(0o600)
API_KEY = os.getenv('FORTE_API_KEY') or KEY_FILE.read_text().strip()
STATIC = Path(__file__).with_name('static')
MAX_BYTES = 100*1024*1024
TTL_SECONDS = 24*3600
jobs = {}
lock = threading.RLock()
state = {'ready':False,'busy':False,'error':None}
connection = process = None


def worker(pipe):
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    try:
        from forte.engine import Engine
        engine=Engine()
        pipe.send(('ready',None))
        while True:
            path=pipe.recv()
            try:
                result=engine.analyze(path,progress=lambda p,s:pipe.send(('progress',{'progress':p,'stage':s})))
                result.pop('_candidates',None)
                pipe.send(('result',result))
            except Exception as exc:
                # Do not leak model output, paths or tracebacks to HTTP clients.
                message = 'Видео не содержит поддерживаемую сцену кассового рабочего места.' if type(exc).__name__=='UnsupportedScene' else 'Видео не удалось полностью обработать. Проверьте формат или повторите запрос.'
                pipe.send(('error',{'code':type(exc).__name__, 'message':message}))
    except Exception as exc:
        pipe.send(('startup_error',type(exc).__name__))


def launch_worker():
    global connection,process
    state.update(ready=False,error=None)
    ctx=mp.get_context('spawn')
    connection,child=ctx.Pipe()
    process=ctx.Process(target=worker,args=(child,),daemon=True)
    process.start()
    if connection.poll(120):
        kind,payload=connection.recv()
        if kind=='ready':state['ready']=True
        else:state['error']='Model initialization failed: '+str(payload)
    else:state['error']='Model initialization timeout'


def save_job(job):
    target=RUNTIME/(job['job_id']+'.json')
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(job,ensure_ascii=False),encoding='utf8')
    temp.replace(target)


def execute_job(job_id,path):
    started=time.monotonic()
    try:
        connection.send(str(path))
        while time.monotonic()-started < 280:
            if not process.is_alive():raise RuntimeError('GPU worker stopped')
            if not connection.poll(0.3):continue
            kind,payload=connection.recv()
            with lock:
                job=jobs[job_id]
                if kind=='progress':job.update(payload)
                elif kind=='result':
                    job.update(payload,status='done',progress=100,stage='Готово');save_job(job);return
                elif kind=='error':
                    job.update(status='failed',error=payload,stage='Ошибка обработки');save_job(job);return
        raise TimeoutError('Processing deadline exceeded')
    except Exception as exc:
        with lock:
            jobs[job_id].update(status='failed',stage='Обработка остановлена',error={'code':type(exc).__name__,'message':'Превышен лимит времени или остановлен GPU-процесс.'})
            save_job(jobs[job_id])
        if process.is_alive():process.terminate();process.join(5)
        launch_worker()
    finally:
        with lock:state['busy']=False


def cleanup():
    now=time.time()
    for path in RUNTIME.iterdir():
        if path.is_file() and path.suffix in ('.mov','.json','.tmp') and now-path.stat().st_mtime>TTL_SECONDS:
            path.unlink(missing_ok=True)
            jobs.pop(path.stem,None)


def retention_loop():
    while True:
        time.sleep(300)
        with lock:cleanup()


@asynccontextmanager
async def lifespan(app):
    for path in RUNTIME.glob('*.json'):
        try:
            job=json.loads(path.read_text(encoding='utf8'))
            if job['status']=='processing':
                job.update(status='failed',error={'code':'Restarted','message':'Сервис перезапущен; отправьте видео повторно.'})
                save_job(job)
            jobs[job['job_id']]=job
        except (ValueError,KeyError):pass
    cleanup()
    threading.Thread(target=retention_loop,daemon=True).start()
    startup=threading.Thread(target=launch_worker,daemon=True);startup.start()
    yield
    if process and process.is_alive():process.terminate();process.join(5)


app=FastAPI(title='ForteVision',version='0.1.0',lifespan=lifespan)
app.mount('/static',StaticFiles(directory=STATIC),name='static')


@app.middleware('http')
async def headers(request:Request,call_next):
    if request.method=='POST':
        try:
            if int(request.headers.get('content-length','0'))>MAX_BYTES+1024*1024:
                return JSONResponse({'detail':'Максимальный размер видео — 100 МБ'},status_code=413)
        except ValueError:
            return JSONResponse({'detail':'Некорректная длина запроса'},status_code=400)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    if request.url.path.startswith('/api'):response.headers['Cache-Control']='no-store'
    return response


def authorize(x_api_key:str|None=Header(default=None)):
    if not x_api_key or not hmac.compare_digest(x_api_key,API_KEY):
        raise HTTPException(401,'Требуется корректный X-API-Key')


@app.get('/',include_in_schema=False)
def index():return FileResponse(STATIC/'index.html')


@app.get('/healthz')
@app.get('/api/v1/health')
def health():return {'status':'ready' if state['ready'] else 'loading','busy':state['busy'],'model':'local'}


@app.get('/api/v1/rules',dependencies=[Depends(authorize)])
def rules():return json.loads(Path(__file__).with_name('rules.json').read_text(encoding='utf8'))


@app.post('/api/v1/analyze',dependencies=[Depends(authorize)],status_code=202)
async def analyze(request:Request,file:UploadFile=File(...)):
    if not (file.filename or '').lower().endswith('.mov'):
        raise HTTPException(415,'Загрузите видео в формате .mov')
    with lock:
        if not state['ready']:raise HTTPException(503,'Модель загружается; повторите запрос через минуту')
        if state['busy']:raise HTTPException(429,'GPU обрабатывает другое видео. Повторите запрос после завершения.',headers={'Retry-After':'15'})
        state['busy']=True
    job_id=uuid.uuid4().hex
    path=RUNTIME/(job_id+'.mov')
    try:
        total=0
        with path.open('wb') as f:
            while chunk:=await file.read(1024*1024):
                total+=len(chunk)
                if total>MAX_BYTES:raise HTTPException(413,'Максимальный размер видео — 100 МБ')
                f.write(chunk)
        if not total:raise HTTPException(422,'Пустой файл')
        # CPU-only validation does not import torch into the web process.
        import av
        try:
            with av.open(str(path)) as c:
                if not c.streams.video:raise ValueError()
                s=c.streams.video[0]
                d=float(s.duration*s.time_base) if s.duration else float(c.duration or 0)/av.time_base
                if not 0<d<=180 or s.width*s.height>3840*2160:raise ValueError()
        except Exception:raise HTTPException(422,'Нужен читаемый MOV длительностью до 180 секунд и разрешением до 4K')
        job={'job_id':job_id,'status':'processing','progress':0,'stage':'Подготовка кадров','created_at':time.time()}
        with lock:jobs[job_id]=job;save_job(job)
        threading.Thread(target=execute_job,args=(job_id,path),daemon=True).start()
        return {'job_id':job_id}
    except BaseException:
        path.unlink(missing_ok=True)
        with lock:state['busy']=False
        raise
    finally:await file.close()


@app.get('/api/v1/jobs/{job_id}',dependencies=[Depends(authorize)])
def get_job(job_id:str):
    if not len(job_id)==32 or any(c not in '0123456789abcdef' for c in job_id):raise HTTPException(404,'Задание не найдено')
    with lock:
        job=jobs.get(job_id)
        if job is None:raise HTTPException(404,'Задание не найдено')
        return job.copy()
