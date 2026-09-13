"""Target-aware, derivative-free search; it never reads simulator outcomes."""
import numpy as np
from .io import fingerprint
DT=6.25e-6
LOW=[45.,.6,.1]
HIGH=[100.,2.,1.5]
LO=np.array(LOW);HI=np.array(HIGH)
def normalized(action):return (np.array(action)-LO)/(HI-LO)
def quantize(unit):
    a=LO+np.asarray(unit)*(HI-LO)
    # Predict and execute identical realizable durations, not unquantized labels.
    a[0]=round(float(a[0]),6)
    a[1:]=np.rint(a[1:]/DT)*DT
    a=np.minimum(HI,np.maximum(LO,a))
    return [float(v) for v in a]
def candidate(unit,kind):
    a=quantize(unit)
    return dict(id='a_'+fingerprint(a)[:16],action=a,kind=kind)
def lhs(n,rng):
    return np.stack([(rng.permutation(n)+rng.random(n))/n for _ in range(3)],1)
def coarse(seed):
    rng=np.random.default_rng(seed)
    corners=np.array([[i,j,k] for i in [0,1] for j in [0,1] for k in [0,1]])
    result=[candidate(u,'global_initial') for u in np.concatenate([corners,lhs(24,rng)])]
    assert len({r['id'] for r in result})==32
    return result
def rank(records,target):
    return sorted(records,key=lambda r:(abs(r['eta_pred']-target),2*r['action'][1]+r['action'][2],r['id']))
def refine(records,target,seed):
    ranked=rank(records,target);centers=[]
    # Prefer well-scored but separated centers in normalized action coordinates.
    # Relax the distance deterministically only if five cannot be found.
    used_threshold=None
    for threshold in [.20,.15,.10,0.]:
        centers=[]
        for r in ranked:
            if all(np.linalg.norm(normalized(r['action'])-normalized(c['action']))>=threshold for c in centers):centers.append(r)
            if len(centers)==5:break
        if len(centers)==5:used_threshold=threshold;break
    assert len(centers)==5
    rng=np.random.default_rng(seed);seen={r['id'] for r in records};new=[]
    for center in centers:
        for scale in [.05,.05,.05,.12,.12,.12]:
            for _ in range(1000):
                u=normalized(center['action'])+rng.normal(0,scale,3)
                u=1-np.abs(np.mod(u,2)-1)  # Reflect instead of clipping onto boundary.
                c=candidate(u,'local');c.update(center_id=center['id'],normalized_scale=scale)
                if c['id'] not in seen:break
            else:raise RuntimeError('Could not draw distinct local action')
            seen.add(c['id']);new.append(c)
    for u in lhs(4,rng):
        c=candidate(u,'global_extra')
        while c['id'] in seen:c=candidate(rng.random(3),'global_extra')
        seen.add(c['id']);new.append(c)
    assert len(new)==34 and len(seen)==66
    return dict(target=target,seed=seed,centers=[r['id'] for r in centers],
        minimum_center_distance=used_threshold,candidates=new)
def tests():
    x=coarse(25000);assert x==coarse(25000) and x!=coarse(25001)
    records=[dict(c,eta_pred=.1+.8*normalized(c['action'])[0]) for c in x]
    r=refine(records,.5,25050);assert r==refine(records,.5,25050)
    assert sum(c['kind']=='local' for c in r['candidates'])==30
    assert sum(c['kind']=='global_extra' for c in r['candidates'])==4
    for c in x+r['candidates']:
        u=normalized(c['action']);assert np.all(u>=-1e-12) and np.all(u<=1+1e-12)
        assert all(abs(round(t/DT)*DT-t)<1e-10 for t in c['action'][1:])
    # A worse refinement can never evict the coarse incumbent.
    assert rank(records+[dict(c,eta_pred=0.) for c in r['candidates']],.5)[0]['id']==rank(records,.5)[0]['id']
    return dict(passed=True,coarse_count=32,local_count=30,extra_count=4,unique_per_task=66)

