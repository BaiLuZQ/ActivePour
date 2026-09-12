"""Recompute planning metrics directly from the published 48 task records."""
import json
import math
from pathlib import Path

rows=json.loads((Path(__file__).resolve().parents[1]/'results/planning.json').read_text())
for label,rr in [('all_computed',rows),('numerically_accepted',[r for r in rows if r['accepted']])]:
    errors=[abs(r['eta_sim']-r['target']) for r in rr]
    print(label,dict(n=len(rr),mae_pp=100*sum(errors)/len(rr),
        rmse_pp=100*math.sqrt(sum(e*e for e in errors)/len(rr)),within_5pp=sum(e<=.05 for e in errors)))
