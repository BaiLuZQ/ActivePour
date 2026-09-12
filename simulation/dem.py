"""Quasi-2D GeoTaichi DEM adapter; numerical convergence is not certified.

Spheres have 3D mass/inertia but motion is constrained to the xy plane.
The adapter supplies deterministic packing, effective-gravity actions and
irreversible rim capture. Contact search, forces and integration are GeoTaichi.
One action per process deliberately avoids incomplete contact-history resets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import shutil

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--geotaichi-root', required=True)
    p.add_argument('--dt', type=float, default=6.25e-6)
    p.add_argument('--radius', type=float, default=.005)
    p.add_argument('--mu', type=float, default=.5)
    p.add_argument('--rolling', type=float, default=.05)
    p.add_argument('--kn', type=float, default=20000.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--theta', type=float, default=65.)
    p.add_argument('--rotation', type=float, default=1.)
    p.add_argument('--hold', type=float, default=.6)
    p.add_argument('--probe', type=float, default=35.)
    p.add_argument('--prep', type=float, default=3.)
    p.add_argument('--rest', type=float, default=3.)
    p.add_argument('--verlet', type=float, default=.2)
    p.add_argument('--jitter-fraction',type=float,default=.01)
    p.add_argument('--post-state',type=Path)
    p.add_argument('--pre-state',type=Path)
    p.add_argument('--pre-only',action='store_true')
    p.add_argument('--probe-only',action='store_true')
    p.add_argument('--probe-rotation',type=float,default=.4)
    p.add_argument('--probe-hold',type=float,default=.6)
    p.add_argument('--rebuild-trigger-scale',type=float,default=1.)
    p.add_argument('--contact-audit',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--arch',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--canonical-order',action='store_true')
    a = p.parse_args()
    assert not (a.pre_state and a.post_state)
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, a.geotaichi_root)
    from geotaichi import DEM, init, ti
    from scipy.spatial import cKDTree
    init(arch=a.arch, cpu_max_num_threads=1, default_fp='float64', device_memory_GB=1,
         offline_cache=True, log=False)
    start = time.perf_counter()
    left, right, floor, rim, z = .25, .625, .25, .5, .025
    rng = np.random.default_rng(a.seed)
    # Fixed lattice spacing guarantees no initial overlap even with radius
    # polydispersity. Settling, not a fabricated final surface, makes the bed.
    step = 2.04*a.radius
    xx, yy = np.meshgrid(np.arange(left+step, right-step/2, step),
                         np.arange(floor+step, floor+.18, step))
    n = xx.size
    rr = a.radius*rng.uniform(.85, 1., n)
    cloud = np.column_stack([xx.ravel(), yy.ravel(), np.full(n,z),rr])
    assert 0<=a.jitter_fraction<=.01
    cloud[:,:2]+=rng.uniform(-a.jitter_fraction,a.jitter_fraction,(n,2))*a.radius
    np.savetxt(a.output/'initial_spheres.txt', cloud)
    d = DEM(log=False)
    d.set_configuration(log=False, domain=ti.Vector([1.,1.,.05]),
                        gravity=[0.,-9.81,0.], engine='SymplecticEuler',
                        search='LinkedCell', visualize=False)
    d.set_solver(log=False, solver={'Timestep':a.dt,'SimulationTime':1.,
                 'SaveInterval':100.,'SavePath':str(a.output/'upstream')})
    d.memory_allocate(log=False,memory={'max_material_number':1,
        'max_particle_number':n,'max_sphere_number':n,'max_plane_number':3,
        'body_coordination_number':32,'wall_coordination_number':3,
        'verlet_distance_multiplier':a.verlet,'compaction_ratio':[1.,1.]})
    d.add_attribute(materialID=0,attribute={'Density':2500.,
                    'ForceLocalDamping':0.,'TorqueLocalDamping':0.})
    d.add_body_from_file(body={'FileType':'TXT','Template':{
        'BodyType':'Sphere','GroupID':0,'MaterialID':0,
        'File':str(a.output/'initial_spheres.txt'),
        'InitialVelocity':[0.,0.,0.],'InitialAngularVelocity':[0.,0.,0.],
        'FixVelocity':['Free','Free','Fix'],
        'FixAngularVelocity':['Fix','Fix','Free']}})
    d.choose_contact_model(particle_particle_contact_model='Linear Rolling Model',
                           particle_wall_contact_model='Linear Rolling Model')
    d.add_property(materialID1=0,materialID2=0,property={
        'NormalStiffness':a.kn,'TangentialStiffness':.5*a.kn,
        'RollingStiffness':.5*a.kn,'TwistingStiffness':.5*a.kn,
        'Friction':a.mu,'RollingFriction':a.rolling,'TwistingFriction':0.,
        'NormalViscousDamping':.3,'TangentialViscousDamping':0.,
        'RollingViscousDamping':0.,'TwistingViscousDamping':0.})
    for center, normal in [([left,0.,z],[1.,0.,0.]),
                            ([right,0.,z],[-1.,0.,0.]),
                            ([0.,floor,z],[0.,1.,0.])]:
        d.add_wall(body={'WallType':'Plane','MaterialID':0,
                        'WallCenter':center,'OuterNormal':normal})
    d.select_save_data(particle=False)
    d.add_essentials({})
    if a.canonical_order:
        assert a.arch=='cpu', 'Canonical accumulation requires the single-thread CPU backend'
        @ti.kernel
        def sort_candidates(potential:ti.template(),prefix:ti.template(),stride:ti.i32):
            # Canonical particle-ID order removes neighbor-cell enumeration
            # order from floating-point force accumulation. No contacts removed.
            for i in range(n):
                start=i*stride
                count=prefix[i+1]-prefix[i]
                for j in range(1,count):
                    value=potential[start+j];k=j-1
                    while k>=0:
                        if potential[start+k]>value:
                            potential[start+k+1]=potential[start+k];k-=1
                        else:break
                    potential[start+k+1]=value
        def ordered_contact_update(model,potential,prefix):
            original=model.update_contact_table
            stride=potential.shape[0]//n
            def update(sims,scene,neighbor):
                sort_candidates(potential,prefix,stride)
                original(sims,scene,neighbor)
            model.update_contact_table=update
        nb=d.contactor.neighbor
        ordered_contact_update(d.contactor.physpp,nb.potential_list_particle_particle,nb.particle_particle)
        ordered_contact_update(d.contactor.physpw,nb.potential_list_particle_wall,nb.particle_wall)
    # Explicit stability scale, including the lightest grain. No silent dt change.
    critical = float(d.get_critical_timestep())
    assert a.dt <= .2*critical, (a.dt, critical)
    d.enginer.pre_calculation(d.sims,d.scene,d.contactor.neighbor)
    assert 0<a.rebuild_trigger_scale<=1.
    # Smaller trigger rebuilds MORE frequently with the same safe search skin.
    d.enginer.limit1*=a.rebuild_trigger_scale**2
    part, sphere = d.scene.particle, d.scene.sphere
    mass = part.m.to_numpy()[:n]
    mass0 = float(mass.sum())
    captured = ti.field(ti.i32,shape=n)

    @ti.kernel
    def capture():
        for i in range(n):
            if captured[i] == 0 and part[i].x[1]-part[i].rad >= rim:
                captured[i] = 1
                part[i].active = 0
                part[i].v = ti.Vector([0.,0.,0.])
                part[i].w = ti.Vector([0.,0.,0.])
                sphere[i].fix_v = ti.Vector([0,0,0])
                sphere[i].fix_w = ti.Vector([0,0,0])
                # Integration in this version does not skip inactive spheres.
                # Park captured grains outside the box, safely inside the domain.
                part[i].x = ti.Vector([.85,.85,z])
                part[i].verletDisp = ti.Vector([1.,0.,0.])

    history, frames = [], {}
    clock = 0

    def inspect(phase):
        x=part.x.to_numpy()[:n]; v=part.v.to_numpy()[:n]
        cc=captured.to_numpy().astype(bool); active=~cc
        speed=np.linalg.norm(v[active],axis=1)
        xa=x[active]; ra=rr[active]
        pairs=np.array(list(cKDTree(xa).query_pairs(2*a.radius)),dtype=int).reshape(-1,2)
        overlap=0.
        if len(pairs):
            distance=np.linalg.norm(xa[pairs[:,0]]-xa[pairs[:,1]],axis=1)
            overlap=float(np.max(np.maximum(0,ra[pairs[:,0]]+ra[pairs[:,1]]-distance)/(2*a.radius)))
        wall=float(max(0.,np.max(ra-(xa[:,0]-left)),
                       np.max(ra-(right-xa[:,0])),np.max(ra-(xa[:,1]-floor)))/(2*a.radius)) if len(xa) else 0.
        result={'t':clock*a.dt,'phase':phase,'eta':float(mass[cc].sum()/mass0),
                'rms':float(np.sqrt(np.mean(speed**2))) if len(speed) else 0.,
                'p99':float(np.quantile(speed,.99)) if len(speed) else 0.,
                'overlap_diameter':overlap,'wall_overlap_diameter':wall,
                'planarity_error':float(np.max(abs(x[:,2]-z))),
                'finite':bool(np.isfinite(x).all() and np.isfinite(v).all()),
                'mass_error':float(abs(mass[cc].sum()+mass[active].sum()-mass0)/mass0)}
        history.append(result)
        (a.output/'history.json').write_text(json.dumps(history,indent=2))
        assert result['finite'] and overlap<.1 and wall<.1, result
        return result

    def save(name):
        r=inspect(name);frames[name]=r
        r['wall_seconds_since_start']=time.perf_counter()-start
        np.savez_compressed(a.output/(name+'.npz'),x=part.x.to_numpy()[:n],
            v=part.v.to_numpy()[:n],w=part.w.to_numpy()[:n],r=rr,m=mass,
            captured=captured.to_numpy())
        print(name,json.dumps(r),flush=True)
        if a.contact_audit:check_contact_rebuild(name)

    def advance(duration,theta0=0.,theta1=0.,phase='wait'):
        nonlocal clock
        steps=round(duration/a.dt)
        assert abs(steps*a.dt-duration)<1e-10
        # Observation hooks never restart the quintic trajectory or advance time.
        # All materials use identical, externally specified observation times.
        observation_times = {
            'probe_out': [(duration*.5,'out_half'), (duration*.75,'out_threequarter'),
                          (duration,'peak_arrival')],
            'probe_hold': [(.1,'peak_hold_010'),(.3,'peak_hold_030')],
            'probe_return': [(duration*.5,'return_half'),(duration,'return_arrival')],
            'probe_rest': [(.1,'rest_010'),(.3,'rest_030'),(1.,'rest_100')],
        }.get(phase, [])
        observations = {round(t/a.dt): name for t,name in observation_times if t<=duration}
        for j in range(steps):
            u=(j+.5)/steps
            smooth=u*u*u*(10+u*(-15+6*u))
            angle=np.deg2rad(theta0+(theta1-theta0)*smooth)
            d.sims.gravity=ti.Vector([9.81*np.sin(angle),-9.81*np.cos(angle),0.])
            d.solver.core(d.scene)
            capture()
            clock+=1
            d.sims.current_step=clock;d.sims.current_time=clock*a.dt
            if clock % round(.1/a.dt)==0: inspect(phase)
            if j+1 in observations: save(observations[j+1])

    # Discover complete public Taichi fields, including tangential/rolling
    # history, neighbor prefixes and Verlet displacements. Never restore x/v only.
    fields={'captured':captured};visited=set()
    def discover(obj,path):
        if id(obj) in visited:return
        visited.add(id(obj))
        if hasattr(obj,'to_numpy') and hasattr(obj,'from_numpy'):
            fields[path]=obj;return
        if type(obj).__module__.startswith('src.'):
            for key,value in vars(obj).items():
                if key not in ['sims','scene']:discover(value,path+'.'+key)
    for key,obj in [('scene',d.scene),('neighbor',d.contactor.neighbor),
                    ('physpp',d.contactor.physpp),('physpw',d.contactor.physpw)]:
        discover(obj,key)
    contact_audits=[]
    def check_contact_rebuild(phase):
        """A zero-time controlled experiment, not two accumulated force steps."""
        backup={key:field.to_numpy() for key,field in fields.items()}
        def restore():
            for key,field in fields.items():field.from_numpy(backup[key])
        d.enginer.reset_particle(d.scene)
        d.enginer.system_resolve(d.sims,d.scene,d.contactor.neighbor)
        f0=part.contact_force.to_numpy();t0=part.contact_torque.to_numpy()
        restore()
        d.enginer.update_verlet_table(d.sims,d.scene,d.contactor.neighbor)
        d.enginer.reset_particle(d.scene)
        d.enginer.system_resolve(d.sims,d.scene,d.contactor.neighbor)
        f1=part.contact_force.to_numpy();t1=part.contact_torque.to_numpy()
        count=int(d.contactor.neighbor.particle_particle.to_numpy()[n])
        cp=d.contactor.physpp.cplist.to_numpy()
        candidates={tuple(sorted((int(i),int(j)))) for i,j in zip(cp['endID1'][:count],cp['endID2'][:count])}
        x=part.x.to_numpy()[:n];active=captured.to_numpy()==0;ids=np.flatnonzero(active)
        pairs=cKDTree(x[active]).query_pairs(2*a.radius)
        actual={tuple(sorted((int(ids[i]),int(ids[j])))) for i,j in pairs
                if np.linalg.norm(x[ids[i]]-x[ids[j]])<rr[ids[i]]+rr[ids[j]]}
        entry={'phase':phase,'force_max_difference_N':float(np.max(abs(f1-f0))),
               'torque_max_difference_Nm':float(np.max(abs(t1-t0))),
               'missed_overlapping_pairs_after_rebuild':len(actual-candidates),
               'duplicate_candidate_pairs_after_rebuild':count-len(candidates)}
        entry['passed']=(entry['force_max_difference_N']<1e-8 and
                         entry['torque_max_difference_Nm']<1e-10 and
                         entry['missed_overlapping_pairs_after_rebuild']==0 and
                         entry['duplicate_candidate_pairs_after_rebuild']==0)
        restore();contact_audits.append(entry)
        (a.output/'contact_audits.json').write_text(json.dumps(contact_audits,indent=2))
        print('CONTACT_REBUILD',json.dumps(entry),flush=True)
    state_keys=['radius','mu','rolling','kn','seed','probe','prep','rest','verlet','jitter_fraction','geotaichi_root',
                'dt','arch','probe_rotation','probe_hold']
    contract={k:getattr(a,k) for k in state_keys}
    if a.canonical_order:contract['canonical_order']=True
    source=Path(a.geotaichi_root)
    source_paths=['src/physics_model/contact_model/LinearRollingModel.py',
           'src/dem/engines/EngineKernel.py','src/dem/contact/ContactKernel.py',
           'src/dem/SceneManager.py','src/dem/generator/BodyGenerator.py']
    source_hashes={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in source_paths}
    pre_contract={k:v for k,v in contract.items() if k not in ['probe','probe_rotation','probe_hold','rest']}
    def snapshot(name, selected_contract):
        arrays={}
        for key,field in fields.items():
            data=field.to_numpy()
            if isinstance(data,dict):arrays.update({key+'::'+sub:value for sub,value in data.items()})
            else:arrays[key]=data
        arrays['metadata']=np.array(json.dumps({'contract':selected_contract,'fields':sorted(fields),
            'time':clock*a.dt,'frames':frames,'history':history,'source_hashes':source_hashes}))
        np.savez_compressed(a.output/(name+'_state.npz'),**arrays)
    def finish():
        report={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'n':n,'mass0':mass0,'critical_dt_scale':critical,'frames':frames,
            'wall_seconds':time.perf_counter()-start,'release_allowed':False,'restore_check':restore_check,
            'source_hashes':source_hashes,
            'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print('COMPLETED',report['wall_seconds'],flush=True)
    restore_check=None
    if a.post_state or a.pre_state:
        selected_state=a.post_state or a.pre_state
        state=np.load(selected_state)
        meta=json.loads(str(state['metadata']))
        expected_contract=contract if a.post_state else pre_contract
        assert meta['contract']==expected_contract, (meta['contract'],expected_contract)
        assert meta['source_hashes']==source_hashes, 'Changed contact/integrator implementation'
        assert sorted(fields)==meta['fields'], 'Snapshot field inventory mismatch'
        for key,field in fields.items():
            # Struct fields export dictionaries: flatten their members below.
            data=field.to_numpy()
            restored=({sub:state[key+'::'+sub] for sub in data} if isinstance(data,dict) else state[key])
            field.from_numpy(restored)
            verify=field.to_numpy()
            if isinstance(verify,dict):
                assert all(np.array_equal(verify[sub],restored[sub]) for sub in verify)
            else:assert np.array_equal(verify,restored)
        clock=round(meta['time']/a.dt)
        assert abs(clock*a.dt-meta['time'])<1e-10
        d.sims.current_step=clock;d.sims.current_time=clock*a.dt
        frames=meta['frames'];history=meta['history']
        for key in frames:
            shutil.copyfile(selected_state.parent/(key+'.npz'),a.output/(key+'.npz'))
        restore_check={'all_fields_exact':True,'field_count':len(fields),
                       'source_snapshot_sha256':hashlib.sha256(selected_state.read_bytes()).hexdigest()}
        if a.contact_audit:check_contact_rebuild('I_post_restored')
        if a.audit_only:
            assert a.contact_audit
            if not all(r['passed'] for r in contact_audits):raise SystemExit(2)
            return
    else:
        advance(a.prep,phase='prep');save('I_pre')
        snapshot('pre',pre_contract)
    if a.pre_only:
        finish();return
    if not a.post_state:
        advance(a.probe_rotation,0.,a.probe,'probe_out')
        advance(a.probe_hold,a.probe,a.probe,'probe_hold');save('I_probe_peak')
        advance(a.probe_rotation,a.probe,0.,'probe_return')
        advance(a.rest,phase='probe_rest');save('I_post')
        snapshot('post',contract)
    if a.probe_only:
        finish();return
    advance(a.rotation,0.,a.theta,'out');save('future_peak')
    advance(a.hold,a.theta,a.theta,'hold');save('future_hold')
    advance(a.rotation,a.theta,0.,'return');save('future_return')
    advance(a.rest,phase='rest');save('future_rest')
    finish()


if __name__=='__main__':
    main()
