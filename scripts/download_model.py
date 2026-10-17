"""One-time preparation only. Inference itself is entirely offline."""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download

root=Path(os.getenv('FORTE_HOME','/home/ubuntu/forte-datathon'))
repo='Qwen/Qwen3-VL-4B-Instruct'
revision=os.getenv('MODEL_REVISION') or HfApi().model_info(repo).sha
snapshot_download(repo_id=repo,revision=revision,local_dir=root/'models'/'Qwen3-VL-4B-Instruct',
    allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*','README.md'],max_workers=4)
(root/'model-manifest.json').write_text(json.dumps({'repository':repo,'revision':revision,'license':'Apache-2.0'},indent=2))
print('Model downloaded and revision recorded in model-manifest.json')
