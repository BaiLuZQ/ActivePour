"""Render five probes and four future frames from a completed DEM run."""
import argparse
from pathlib import Path
from .render import raw_particles,assemble

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    probe=['I_pre','peak_arrival','I_probe_peak','return_arrival','I_post']
    future=['future_peak','future_hold','future_return','future_rest']
    for key in probe:raw_particles(root/(key+'.npz'),256).save(out/(key+'.png'))
    assemble({k:raw_particles(root/(k+'.npz'),256) for k in future},future,2).save(out/'future.png')

if __name__=='__main__':main()
