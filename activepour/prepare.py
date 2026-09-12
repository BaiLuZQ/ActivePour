"""Prepare portable tensor banks from an explicit, group-split image manifest."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .io import read,save,sha,safe_path
from activepour_model.data import image_tensor,encode_latent

PROMPT=('Scientific grayscale simulation of many small black spherical grains on a plain white background. '
        'Four panels in a two by two grid, fixed container coordinates. Top left maximum tilt arrival, '
        'top right hold end, bottom left upright return end, bottom right settled state. '
        'Sharp individual particles and empty white space.')

def validate(rows):
    seen=set();groups={}
    for r in rows:
        if r['id'] in seen:raise ValueError('Duplicate sample ID')
        seen.add(r['id'])
        if r['split'] not in ['train','validation','test']:raise ValueError('Invalid split')
        if groups.setdefault(r['episode_id'],r['split'])!=r['split']:raise ValueError('Probe episode crosses splits')
        if len(r['probe_paths'])!=5 or len(r['action'])!=3:raise ValueError('Expected five probes and three actions')
        if not 0<=r['eta']<=1:raise ValueError('Invalid discharge fraction')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();rows=read(a.manifest);validate(rows)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from diffusers import StableDiffusion3Pipeline,AutoencoderKL
    pipe=StableDiffusion3Pipeline.from_pretrained(a.model,transformer=None,vae=None,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda')
    with torch.inference_mode():
        text,_,pooled,_=pipe.encode_prompt(prompt=PROMPT,prompt_2=PROMPT,prompt_3=PROMPT,
            max_sequence_length=128,do_classifier_free_guidance=False,device='cuda')
    text=text.cpu();pooled=pooled.cpu();del pipe;torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(a.model,subfolder='vae',local_files_only=True,torch_dtype=torch.float32).cuda().eval()
    for split in ['train','validation','test']:
        rr=[r for r in rows if r['split']==split]
        if not rr:continue
        probes=[];targets=[];union=np.zeros((512,512),dtype=bool)
        for r in rr:
            pp=[]
            for path in r['probe_paths']:
                with Image.open(safe_path(a.root,path)) as im:arr=np.asarray(im.convert('L'),dtype=np.float32)
                if arr.shape!=(256,256):raise ValueError('Probe images must be 256 x 256')
                pp.append(arr/127.5-1)
            probes.append(torch.from_numpy(np.stack(pp)))
            image=image_tensor(safe_path(a.root,r['future']))
            if image.shape!=(1,3,512,512):raise ValueError('Future grid must be 512 x 512')
            if split=='train':union|=((1-image.mean(1)[0].numpy())/2>.2)
            with torch.inference_mode():targets.append(encode_latent(vae,image.cuda()).cpu())
        bank=dict(split=split,ids=[r['id'] for r in rr],episode_ids=[r['episode_id'] for r in rr],
                  probe=torch.stack(probes),action=torch.tensor([r['action'] for r in rr]),
                  eta=torch.tensor([r['eta'] for r in rr]),target=torch.cat(targets),
                  text_tokens=text,pooled_text=pooled,manifest_sha256=sha(a.manifest),prompt=PROMPT)
        if split=='train':bank['foreground_union']=torch.from_numpy(union)[None,None]
        save(out/(split+'.pt'),bank)
    print('Prepared tensor banks. Only training data may fit normalization or loss weights.')

if __name__=='__main__':main()
