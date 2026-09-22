#!/usr/bin/env python3
"""Public checkpoint acquisition; transport mirror, no credentials or paid API."""
import json
import os
from pathlib import Path
import time
from huggingface_hub import snapshot_download

out=Path('/home/qiankun/phd_research/vqa_semcom/outputs/accuracy_upgrade_20260908')
begin=time.time()
status={'model':'Qwen/Qwen2.5-VL-7B-Instruct','endpoint':os.environ.get('HF_ENDPOINT'),
        'started_unix':begin,'pid':os.getpid(),'state':'DOWNLOADING'}
(out/'download_status.json').write_text(json.dumps(status,indent=2))
try:
    path=snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct',max_workers=4,
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],etag_timeout=15)
    status.update(state='COMPLETE',path=path,seconds=time.time()-begin)
except Exception as exc:
    status.update(state='BLOCKED',error=repr(exc),seconds=time.time()-begin)
    raise
finally:
    (out/'download_status.json').write_text(json.dumps(status,indent=2))
