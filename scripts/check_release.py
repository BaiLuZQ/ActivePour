"""Static release checks: English source, no weights, no obvious credentials."""
import ast
import re
from pathlib import Path

root=Path(__file__).resolve().parents[1]
errors=[]
for p in root.rglob('*'):
    if not p.is_file() or any(x in p.parts for x in ['.git','__pycache__','.venv']):continue
    if any(x.endswith('.egg-info') for x in p.parts):continue
    if p.suffix in ['.pt','.pth','.safetensors','.npz','.pem','.key']:errors.append((str(p.relative_to(root)),'excluded artifact'))
    if p.suffix not in ['.py','.md','.json','.toml','.txt','.yml']:continue
    text=p.read_text(encoding='utf-8')
    if re.search(r'[\u4e00-\u9fff]',text):errors.append((str(p.relative_to(root)),'non-English source/text'))
    patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'hf_[A-Za-z0-9]{20,}',
              r'-----BEGIN [A-Z ]*PRIVATE KEY-----',r'/home/[A-Za-z0-9_-]+/',r'[A-Z]:[\\/](?:Users|world_diffusion)[\\/]']
    for pattern in patterns:
        if re.search(pattern,text):errors.append((str(p.relative_to(root)),'potential credential/private path'))
    if p.suffix=='.py':ast.parse(text,filename=str(p))
if errors:raise SystemExit(str(errors))
print('Static checks passed: parsable English source, no obvious secrets or private paths.')
