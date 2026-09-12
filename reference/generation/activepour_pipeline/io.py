"""Atomic publications: pointers are updated only after payloads are durable."""
import hashlib
import json
import os
from pathlib import Path
import uuid


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def inside(root,relative):
    root=Path(root).resolve(); p=(root/relative).resolve()
    if not p.is_relative_to(root) or p==root:raise ValueError('path escapes dataset')
    return p
