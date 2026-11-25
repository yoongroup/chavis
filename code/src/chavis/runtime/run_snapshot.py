from __future__ import annotations
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .hashing import sha256_file
SCHEMA = 'chavis_run_snapshot_v1'
DEFAULT_CODE = (
    'src',
    'tools',
    'config',
    'campaigns/ni_drm/config',
    'support/runtime_support/automation_search'
)
SKIP_PARTS = {'__pycache__', '.pytest_cache', 'runs', 'slab_store'}
SKIP_SUFFIX = ('.pyc', '.orig', '.log')
PACKAGES = ('numpy', 'scipy', 'ase', 'torch', 'fairchem-core', 'e3nn', 'pymatgen')


def _sha(path, chunk=1 << 20):
    return sha256_file(path, chunk_size=chunk)


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def _code_files(code_root, entries):
    root = Path(code_root).resolve()
    out = []
    for e in entries:
        p = root / e if not Path(e).is_absolute() else Path(e)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob('*')):
                if f.is_file() and (not set(f.relative_to(root).parts) & SKIP_PARTS) and (not f.name.endswith(SKIP_SUFFIX)):
                    out.append(f)
    return (root, sorted(set(out)))


def _rel(root, f):
    try:
        return f.resolve().relative_to(root).as_posix()
    except ValueError:
        h = hashlib.sha256(str(f.resolve()).encode()).hexdigest()[:10]
        return f'_external/{h}_{f.name}'


def _environment():
    from importlib import metadata
    pk = {}
    for name in PACKAGES:
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    gpu = cuda_driver = None
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )
        gpu = r.stdout.strip() or None
        h = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=10
        ).stdout
        if 'CUDA Version:' in h:
            cuda_driver = h.split('CUDA Version:')[1].split()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    torch_cuda = None
    if 'torch' in sys.modules:
        torch_cuda = getattr(sys.modules['torch'].version, 'cuda', None)
    return {
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'platform': platform.platform(),
        'packages': pk,
        'gpu_nvidia_smi': gpu,
        'cuda_driver_version': cuda_driver,
        'torch_cuda_build': torch_cuda,
        'cuda_env': {k: os.environ.get(k) for k in (
            'CUDA_HOME',
            'CUDA_PATH',
            'LD_LIBRARY_PATH'
        )},
        'threads': {k: os.environ.get(k) for k in (
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS'
        )},
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')
    }


def _inputs(paths, listing_dir=None):
    out = []
    for k, p in enumerate(paths or []):
        p = Path(p)
        if p.is_file():
            out.append({
                'path': str(p),
                'kind': 'file',
                'sha256': _sha(p),
                'bytes': p.stat().st_size
            })
        elif p.is_dir():
            files = sorted((f for f in p.rglob('*') if f.is_file()))
            listing = ('\n'.join((f'{_sha(f)} *{f.relative_to(p).as_posix()}' for f in files)) + '\n').encode('utf-8')
            entry = {
                'path': str(p),
                'kind': 'dir',
                'n_files': len(files),
                'listing_sha256': hashlib.sha256(listing).hexdigest()
            }
            if listing_dir is not None:
                lf = Path(listing_dir) / f'input_{k:02d}_listing.txt'
                lf.parent.mkdir(parents=True, exist_ok=True)
                lf.write_bytes(listing)
                entry['listing_file'] = lf.name
            out.append(entry)
        else:
            out.append({'path': str(p), 'kind': 'missing'})
    return out


def start_run(
    attempt_dir,
    code_root,
    argv,
    config=None,
    input_paths=(),
    model=None,
    workers=None,
    extra_code=(),
    code_entries=DEFAULT_CODE,
    parent_attempt=None,
    note=None
):
    snap = Path(attempt_dir) / '_snapshot'
    if snap.exists():
        raise FileExistsError(f'{snap} already exists: use a new attempt directory for every run/shard/restart')
    (snap / 'code').mkdir(parents=True)
    root, files = _code_files(
        code_root,
        list(code_entries) + [str(Path(x).resolve()) for x in extra_code]
    )
    lines = []
    for f in files:
        rel = _rel(root, f)
        dst = snap / 'code' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        lines.append(f'{_sha(dst)} *{rel}')
    manifest = ('\n'.join(lines) + '\n').encode('utf-8')
    (snap / 'code_manifest.txt').write_bytes(manifest)
    m = dict(model or {})
    ck = m.get('checkpoint')
    if ck and Path(ck).is_file():
        m['checkpoint_sha256'] = _sha(ck)
    elif m.get('used', True) is False:
        m['checkpoint_sha256'] = 'not_used'
    else:
        m['checkpoint_sha256'] = None
        m.setdefault(
            'checkpoint_note',
            'unknown: checkpoint path not given or not found'
        )
    rec = {
        'schema': SCHEMA,
        'start': _now(),
        'end': None,
        'status': 'running',
        'error': None,
        'argv': list(argv),
        'cwd': os.getcwd(),
        'code_root': str(root),
        'code_manifest_sha256': _sha(snap / 'code_manifest.txt'),
        'n_code_files': len(files),
        'code_sources': {_rel(root, f): str(f) for f in files},
        'config': config,
        'inputs': _inputs(input_paths, snap / 'inputs'),
        'model': m,
        'workers': workers,
        'environment': _environment(),
        'parent_attempt': parent_attempt,
        'note': note
    }
    (snap / 'run.json').write_bytes(json.dumps(rec, indent=1, default=str).encode('utf-8'))
    return snap / 'run.json'


def end_run(attempt_dir, status, error=None):
    snap = Path(attempt_dir) / '_snapshot'
    rec = json.loads((snap / 'run.json').read_bytes().decode('utf-8'))
    expected = dict((l.split(' *', 1)[::-1] for l in (snap / 'code_manifest.txt').read_bytes().decode('utf-8').splitlines() if l))
    changed = []
    for rel, src in rec['code_sources'].items():
        if not Path(src).is_file() or _sha(src) != expected.get(rel):
            changed.append(rel)
    rec.update(
        end=_now(),
        status=status,
        error=error,
        code_changed_during_run=changed,
        reproducible_code=not changed
    )
    (snap / 'run.json').write_bytes(json.dumps(rec, indent=1, default=str).encode('utf-8'))
    return rec
