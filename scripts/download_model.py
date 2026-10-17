"""One-time preparation only. Inference itself is entirely offline."""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
repo=os.getenv('MODEL_ID','Qwen/Qwen3-VL-4B-Instruct')
if repo not in {'Qwen/Qwen3-VL-4B-Instruct','Qwen/Qwen3-VL-8B-Instruct'}:raise ValueError('Review model license before adding another model')
target=root/'models'/repo.split('/')[-1]
revision=os.getenv('MODEL_REVISION') or HfApi().model_info(repo).sha
snapshot_download(repo_id=repo,revision=revision,local_dir=target,
    allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*','README.md'],max_workers=4)
manifest=json.dumps({'repository':repo,'revision':revision,'license':'Apache-2.0'},indent=2)
(target/'model-manifest.json').write_text(manifest)
if repo=='Qwen/Qwen3-VL-4B-Instruct':(root/'model-manifest.json').write_text(manifest)
print('Model downloaded and revision recorded in model-manifest.json')
