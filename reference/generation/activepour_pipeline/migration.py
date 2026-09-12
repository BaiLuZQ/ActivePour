"""Explicit, narrow v1 post-only -> v2 dual-branch migration (not resume)."""
from .io import sha


def migrate_post_parent(state, path, expected_hash, compatible, conditioner):
    if sha(path) != expected_hash or state['format'] != 'activepour_resume_v1':
        raise ValueError('migration parent hash/format mismatch')
    old = state['compatible']; new = dict(compatible)
    if new['conditioner_spec'].get('architecture')=='temporal_probe_v1':
        for key in ['base','vae','prompt','max_text_length','lora_rank','lora_alpha','lora_modules']:
            if old[key]!=new[key]:raise ValueError('Parent compatibility mismatch: '+key)
        if old['condition_mode']!='post1':raise ValueError('Use common post-only parent')
        current=conditioner.state_dict();source=state['conditioner']
        mapped={}
        for name,value in source.items():
            if name not in current:raise ValueError('Unexpected legacy static parameter '+name)
            if name=='probe_encoder.0.weight':
                # Old post1 input was [post,0,0]. One-channel shared CNN therefore
                # starts from exactly its post-channel kernel, not a channel average.
                if value.shape[1]!=3:raise ValueError('Old input kernel shape')
                value=value[:,:1].clone()
            if current[name].shape!=value.shape:raise ValueError('Shape mismatch '+name)
            mapped[name]=value
        result=conditioner.load_state_dict(mapped,strict=False)
        if result.unexpected_keys or any(not (n.startswith('response_') or n=='observation_seconds') for n in result.missing_keys):
            raise ValueError('Unexpected temporal migration keys')
        return result.missing_keys
    spec = dict(new['conditioner_spec'])
    if spec.pop('architecture', None) != 'post_plus_response_v1':
        raise ValueError('migration target must be dual branch')
    new['conditioner_spec'] = spec; new['condition_mode'] = 'post1'
    if new != old:
        raise ValueError('migration changes more than architecture/input mode')
    current = conditioner.state_dict(); source = state['conditioner']
    additions = set(current)-set(source)
    if set(source)-set(current) or not additions or any(not k.startswith('response_') for k in additions):
        raise ValueError('unexpected state mapping')
    for k, value in source.items():
        if current[k].shape != value.shape:
            raise ValueError('migration shape mismatch: '+k)
    result = conditioner.load_state_dict(source, strict=False)
    if set(result.missing_keys) != additions or result.unexpected_keys:
        raise ValueError('migration load mismatch')
    return sorted(additions)
