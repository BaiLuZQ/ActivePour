"""General release reader and per-image target cache; post1 never reads pre/peak PNG."""
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from .io import sha,fingerprint,atomic_json,read
from .dataset import load_release


def probe_tensor(root,row,mode):
    paths=row['inputs']['probe_paths']
    indices={'pre1':[0],'full5':[0,1,2,3,4]}[mode]
    if len(paths)!=5:raise ValueError('five-frame release required')
    bank=np.zeros((5,256,256),dtype=np.float32)
    # Read ONLY permitted files. Padding is not a fabricated observation.
    for i in indices:
        with Image.open(Path(root)/paths[i]) as image:
            array=np.asarray(image.convert('L'),dtype=np.float32)
        if array.shape!=(256,256):raise ValueError('probe shape')
        bank[i]=array/127.5-1
    return torch.from_numpy(bank)


class TargetCache:
    def __init__(self,root,vae_hash):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.vae_hash=vae_hash

    def key(self,image):
        return fingerprint({'image_sha256':sha(image),'vae_sha256':self.vae_hash,
                            'encoding':'rgb_minus1to1_mode_scaled_shifted_v1'})

    def get(self,image,encode=None):
        key=self.key(image);p=self.root/(key+'.pt');receipt=self.root/(key+'.json')
        if receipt.exists():
            if sha(p)!=read(receipt)['sha256']:raise ValueError('target cache corrupted')
            return torch.load(p,map_location='cpu',weights_only=True),True
        if encode is None:raise FileNotFoundError('latent missing, no encoder provided')
        # Caller owns one cache-population process; training only reads receipts.
        z=encode(image).detach().cpu();tmp=p.with_suffix('.pending')
        if tmp.exists() or p.exists():raise FileExistsError('unfinished cache entry: '+key)
        with tmp.open('xb') as f:
            import os
            torch.save(z,f);f.flush();os.fsync(f.fileno())
        tmp.replace(p);atomic_json(receipt,{'sha256':sha(p),'vae_sha256':self.vae_hash,'image_sha256':sha(image)})
        return z,False


def prepare_cache(root,release,cache_root,legacy_cache,legacy_dataset,mode,split='train',encode=None,row_ids=None):
    # Verified old target latents can be reused independently of new sample names.
    rows=load_release(root,release,split)
    if row_ids is not None:
        wanted=set(row_ids);rows=[r for r in rows if r['id'] in wanted]
        if len(rows)!=len(wanted):raise ValueError('evaluation subset contains absent IDs')
    if not rows:raise ValueError('empty split')
    legacy=Path(legacy_cache);report=read(legacy/'cache_report.json')
    if sha(legacy/'cache.pt')!=report['cache_sha256']:raise ValueError('legacy cache hash')
    previous=torch.load(legacy/'cache.pt',map_location='cpu',weights_only=True)
    oldrows=read(legacy/'selection.json');hashes=read(legacy/'input_hashes.json')
    indexed={hashes[r['targets']['future_grid']]:previous['target_latents'][i:i+1] for i,r in enumerate(oldrows)}
    cache=TargetCache(cache_root,report['vae_weight_sha256']);hits=imports=encoded=0;targets=[]
    for row in rows:
        image=Path(root)/row['targets']['future_grid'];h=sha(image)
        def compute(path):
            nonlocal imports,encoded
            if h in indexed:imports+=1;return indexed[h]
            if encode is None:raise FileNotFoundError('new target needs VAE encoding')
            encoded+=1;return encode(path)
        z,hit=cache.get(image,compute);hits+=int(hit);targets.append(z)
    data={'target_latents':torch.cat(targets),'probe_features':torch.stack([probe_tensor(root,r,mode) for r in rows]),
          'actions':torch.tensor([r['inputs']['action_deg_s_s'] for r in rows]),
          'text_tokens':previous['text_tokens'],'pooled_text':previous['pooled_text'],
          'ids':[r['id'] for r in rows],'probe_mode':'raw'}
    stats={'cache_hits':hits,'imported_legacy_latents':imports,'vae_encoded':encoded,'rows':len(rows),
           'split':split,'condition_mode':mode,'release_sha256':sha(release),
           'base_transformer_sha256':report['transformer_weight_sha256'],'vae_sha256':report['vae_weight_sha256'],
           'prompt':report['prompt'],'max_text_length':report['max_text_length'],
           'fixed_text_source_cache_sha256':report['cache_sha256']}
    return data,rows,stats
