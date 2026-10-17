"""Extract the user-provided archive safely, verify manifest, no video outputs."""
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
dest=root/'data'
dest.mkdir(exist_ok=True)
archive=Path('/home/ubuntu/open.zip')
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        if member.is_dir():continue
        name=member.filename
        if not member.flag_bits&0x800:
            try:name=name.encode('cp437').decode('utf8')
            except (UnicodeError,LookupError):pass
        name=name.replace('\\','/')
        target=(dest/name).resolve()
        if not target.is_relative_to(dest.resolve()):raise ValueError('Unsafe archive path')
        if member.file_size>150*1024*1024:raise ValueError('Unexpected member size')
        target.parent.mkdir(parents=True,exist_ok=True)
        with z.open(member) as source,target.open('wb') as out:shutil.copyfileobj(source,out)
audit=json.loads((dest/'audit.json').read_text(encoding='utf-8-sig'))
paths={p.name:p for p in dest.rglob('*.mov')}
for record in audit['records']:
    path=paths[record['name']]
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=record['sha256']:raise ValueError('Dataset checksum mismatch')
print(json.dumps({'verified_videos':len(audit['records']),'unique_videos':len({r['sha256'] for r in audit['records']}),'bytes':sum(p.stat().st_size for p in paths.values())}),flush=True)
