"""Verify and extract completed ZIP entries while a large upload is ongoing.

Offsets come from a manifest generated from the original archive, never guessed.
Only completed checksum-verified videos become visible to evaluation.
"""
import hashlib
import json
import os
import struct
import time
import zlib
from pathlib import Path

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
index=json.loads((root/'private/zip-index.json').read_text())
audit=json.loads((root/'data/audit.json').read_text(encoding='utf-8-sig'))
hashes={r['name']:r['sha256'] for r in audit['records']}
dest=root/'data'/'videos';dest.mkdir(parents=True,exist_ok=True)
archive=Path('/home/ubuntu/open.zip')
while True:
    count=0
    for item in index:
        out=dest/item['name']
        if out.exists():count+=1;continue
        with archive.open('rb') as f:
            f.seek(item['offset']);header=f.read(30)
            if len(header)<30:continue
            if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP entry')
            name_len,extra_len=struct.unpack_from('<HH',header,26)
            start=item['offset']+30+name_len+extra_len
            if archive.stat().st_size<start+item['size']:continue
            f.seek(start);compressed=f.read(item['size'])
        raw=zlib.decompress(compressed,-15) if item['method']==8 else compressed
        if hashlib.sha256(raw).hexdigest()!=hashes[item['name']]:raise ValueError('Checksum mismatch')
        temp=out.with_suffix('.part');temp.write_bytes(raw);temp.replace(out);count+=1
    print(json.dumps({'verified_ready':count,'total':len(index)}),flush=True)
    if count==len(index):break
    time.sleep(10)
