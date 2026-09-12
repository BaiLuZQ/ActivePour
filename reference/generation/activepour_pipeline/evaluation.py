"""Checkpoint-linked future generation with per-sample resumable publications.

Only a verified sample receipt commits its PNG, latent and metrics. An interrupted
sample may be recomputed; committed samples are never sampled again. The public
evaluate() API and evaluation.json/latent formats remain compatible with node9.
"""
import os
import socket
import uuid
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import SD3Transformer2DModel, AutoencoderKL
from activepour_model.conditioning import PhysicsConditioner
from activepour_model.lora import attach_lora, load_lora_state
from activepour_model.training import sample, metrics_for
from activepour_model.data import encode_latent, image_tensor, decode_latent, to_pixels
from .checkpoint import load
from .dataset import load_release
from .io import read, sha, atomic_json, fingerprint
from .model_data import prepare_cache, probe_tensor


def _sample_identity(identity_sha, cp_hash, release_sha, steps, seed, sid):
    return dict(evaluation_identity_sha256=identity_sha, checkpoint_sha256=cp_hash,
                dataset_release_sha256=release_sha, sampling_steps=steps,
                seed=seed, sample_id=sid)


def _verified_sample(output, name, expected):
    receipt = output / 'sample_receipts' / (name + '.json')
    if not receipt.exists():
        return None
    committed = read(receipt)
    if committed['identity'] != expected:
        raise ValueError('evaluation sample identity changed: ' + name)
    expected_files = {'png': name + '.png', 'predicted_latent': name + '_predicted_latent.pt'}
    result = committed['result']
    for key, filename in expected_files.items():
        if result[key] != filename or sha(output / filename) != committed['files_sha256'][key]:
            raise ValueError('evaluation sample payload mismatch: ' + name)
    if any(result[k] != expected[k] for k in ['sample_id', 'checkpoint_sha256', 'seed']):
        raise ValueError('evaluation result provenance mismatch: ' + name)
    if committed['result_sha256'] != fingerprint(result):
        raise ValueError('evaluation result metrics mismatch: ' + name)
    return result


def _atomic_payload(path, writer):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _preserve_uncommitted(output, name):
    # A crash between payloads and the receipt is not a committed sample. Keep
    # these exact, scoped files for inspection before publishing its new attempt.
    for suffix in ['.png', '_predicted_latent.pt']:
        path = output / (name + suffix)
        if path.exists():
            abandoned = output / 'uncommitted'
            abandoned.mkdir(exist_ok=True)
            path.rename(abandoned / (path.name + '.' + uuid.uuid4().hex))


@torch.no_grad()
def evaluate(run, output, split, *, limit=None, steps=None, which='latest',
             verify_isolation=True, seed_override=None, row_ids=None):
    run = Path(run); output = Path(output)
    manifest = read(run / 'run_manifest.json'); config = dict(manifest['config'])
    state, cp = load(run, manifest['identity'], which=which); cp_hash = sha(cp)
    if sha(Path(config['model']) / 'transformer/diffusion_pytorch_model.safetensors') != state['compatible']['base']:
        raise ValueError('base model changed')
    if sha(Path(config['model']) / 'vae/diffusion_pytorch_model.safetensors') != state['compatible']['vae']:
        raise ValueError('VAE changed')
    seed = config['evaluation_seed'] if seed_override is None else seed_override
    config['sample_steps'] = steps or config['sample_steps']
    release_sha = sha(config['release'])
    rows = load_release(config['dataset'], config['release'], split)
    if row_ids is not None:
        wanted = set(row_ids); rows = [row for row in rows if row['id'] in wanted]
        if len(rows) != len(wanted):
            raise ValueError('evaluation subset contains absent IDs')
    selected_rows = rows[:limit] if limit else rows
    if not selected_rows:
        raise ValueError('empty evaluation selection')
    identity = dict(format='activepour_evaluation_resume_v1', run=str(run.resolve()),
        run_identity_sha256=fingerprint(manifest['identity']), checkpoint_sha256=cp_hash,
        checkpoint_selection=which, dataset_release_sha256=release_sha, split=split,
        condition_mode=config['condition_mode'], sampling_steps=config['sample_steps'],
        seed=seed, guidance_scale=1., verify_isolation=verify_isolation,
        config_sha256=fingerprint(config), evaluation_source_sha256=sha(__file__),
        sample_ids=[row['id'] for row in selected_rows])
    identity_sha = fingerprint(identity)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'evaluation.lock'; token = uuid.uuid4().hex
    with lock.open('x') as stream:
        # Hard-kill recovery must verify this PID/host before clearing a stale lock.
        import json
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(), token=token), stream)
        stream.flush(); os.fsync(stream.fileno())
    try:
        identity_path = output / 'evaluation_identity.json'
        if identity_path.exists():
            if read(identity_path) != identity:
                raise ValueError('evaluation output belongs to another identity')
        else:
            if any(path != lock for path in output.iterdir()):
                raise ValueError('unidentified pre-existing evaluation output; manual review required')
            atomic_json(identity_path, identity)
        completed = {}
        for row in selected_rows:
            name = row['id'] + f'_noise{seed}'
            expected = _sample_identity(identity_sha, cp_hash, release_sha, config['sample_steps'], seed, row['id'])
            result = _verified_sample(output, name, expected)
            if result is not None:
                completed[row['id']] = result
        stats_path = output / 'cache_report.json'
        stats = read(stats_path) if stats_path.exists() else None
        pending = len(selected_rows) - len(completed)
        print('EVALUATION_RESUME', split, 'committed', len(completed), 'pending', pending, flush=True)
        if pending:
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
            vae = AutoencoderKL.from_pretrained(config['model'], subfolder='vae', local_files_only=True,
                torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
            def encode_missing(path):
                if config.get('read_only_cache'):
                    raise RuntimeError('existing target cache required; evaluation will not re-encode')
                return encode_latent(vae, image_tensor(path).cuda()).cpu()
            cache, cache_rows, current_stats = prepare_cache(config['dataset'], config['release'],
                config['target_cache'], config['legacy_cache'], config['legacy_dataset'],
                config['condition_mode'], split=split, encode=encode_missing, row_ids=row_ids)
            if [row['id'] for row in cache_rows] != [row['id'] for row in rows]:
                raise ValueError('evaluation cache row order changed')
            # Retain first-attempt cache accounting; hits may change on resumption.
            if stats is None:
                stats = current_stats; atomic_json(stats_path, stats)
            transformer = SD3Transformer2DModel.from_pretrained(config['model'], subfolder='transformer',
                local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
            attach_lora(transformer, config['lora_rank'], config['lora_alpha'])
            load_lora_state(transformer, state['lora']); transformer.requires_grad_(False)
            conditioner = PhysicsConditioner(**state['compatible']['conditioner_spec']).cuda().eval()
            conditioner.load_state_dict(state['conditioner']); conditioner.requires_grad_(False)
            for i, row in enumerate(selected_rows):
                if row['id'] in completed:
                    continue
                name = row['id'] + f'_noise{seed}'
                expected = _sample_identity(identity_sha, cp_hash, release_sha, config['sample_steps'], seed, row['id'])
                # Only observed probes/actions/text enter sampling, never labels.
                condition = {key: cache[key][i:i+1].cuda() for key in ['probe_features', 'actions']}
                condition.update(text_tokens=cache['text_tokens'].cuda(), pooled_text=cache['pooled_text'].cuda())
                z = sample(transformer, conditioner, condition, config, 0, seed)
                if z.shape != (1, 16, 64, 64) or not torch.isfinite(z).all():
                    raise FloatingPointError('invalid sampled future latent')
                y = to_pixels(decode_latent(vae, z.float()))[0]
                if not np.isfinite(y).all():
                    raise FloatingPointError('nonfinite decoded future image')
                truth = to_pixels(image_tensor(Path(config['dataset']) / row['targets']['future_grid']))[0]
                result = dict(sample_id=row['id'], checkpoint_sha256=cp_hash, seed=seed,
                    png=name + '.png', predicted_latent=name + '_predicted_latent.pt',
                    metrics=metrics_for(config, truth, y))
                if config['condition_mode'] == 'post1' and verify_isolation:
                    changed = dict(row, inputs=dict(row['inputs'], probe_paths=['NONEXISTENT'] * 4 + [row['inputs']['probe_paths'][4]]))
                    alternate = dict(condition, probe_features=probe_tensor(config['dataset'], changed, 'post1')[None].cuda())
                    z2 = sample(transformer, conditioner, alternate, config, 0, seed)
                    result['post1_ignores_pre_peak_exact'] = torch.equal(z, z2)
                _preserve_uncommitted(output, name)
                _atomic_payload(output / result['predicted_latent'], lambda stream: torch.save(z.cpu(), stream))
                pixels = Image.fromarray(np.rint(y * 255).astype('uint8'))
                _atomic_payload(output / result['png'], lambda stream: pixels.save(stream, format='PNG'))
                atomic_json(output / 'sample_receipts' / (name + '.json'),
                    dict(identity=expected, result=result, result_sha256=fingerprint(result),
                         files_sha256={key: sha(output / result[key]) for key in ['png', 'predicted_latent']}))
                completed[row['id']] = result
                print('EVALUATED', split, row['id'], result['metrics']['mean_iou'], flush=True)
        if stats is None:
            raise ValueError('committed evaluation samples have no cache provenance')
        results = [completed[row['id']] for row in selected_rows]
        atomic_json(output / 'evaluation.json', dict(run_id=run.name, checkpoint_sha256=cp_hash,
            checkpoint_selection=which, dataset_release_sha256=release_sha, split=split,
            condition_mode=config['condition_mode'], sampling_steps=config['sample_steps'],
            guidance_scale=1., cache_report=stats, results=results, scope=config['scope'],
            short_steps_override=steps, evaluation_identity_sha256=identity_sha,
            resume_format='activepour_evaluation_resume_v1'))
    finally:
        if lock.exists() and read(lock)['token'] == token:
            lock.unlink()
