from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get('CHAVIS_RUN_ROOT', Path.home() / 'chavis_runs'))
PYTHON = Path(os.environ.get('CHAVIS_PYTHON', sys.executable))
HF_CACHE = Path(os.environ.get('HF_HUB_CACHE',
                Path(os.environ.get('HF_HOME',
                     Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'huggingface')) / 'hub'))
REVISION = 'f611b917d9c68566bbbeccbb0aa0f7cad1696cb2'
PAPER15 = 'D01 D02 D03 D04 D05 D06 D07 D08 D09 D10 D13 N01 N02 N03 N04'.split()
CODE_ENTRIES = ('src', 'tools', 'config', 'campaigns/ni_drm/config', 'support/runtime_support',
                'ultra_fast_v1', 'generic_v2', 'reproduce/drivers', 'reproduce/inputs/reference',
                'reproduce/inputs/ultra_fast', 'reproduce/inputs/cohorts', 'environment',
                'launch_general_v2.sh', 'launch_ultra_fast.sh')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as fh:
        return json.load(fh)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_fingerprint(root):
    """Identify code for resume; no stored build manifest or integrity check."""
    root = Path(root)
    files = []
    for entry in (*CODE_ENTRIES, 'slab_store'):
        path = root / entry
        files.extend(path.rglob('*') if path.is_dir() else [path])
    return digest_json({p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)
                        if p.is_file() and '__pycache__' not in p.parts
                        and p.suffix not in ('.pyc', '.log')})


def cohort(value='paper15', root=ROOT):
    path = Path(root) / 'reproduce/inputs/cohorts' / (value + '.json') if value in ('paper15',) else Path(value)
    doc = read_json(path)
    ids = doc if isinstance(doc, list) else doc.get('reactions')
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        raise ValueError('Cohort must contain a nonempty reactions list of exact IDs')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate reaction IDs in cohort')
    registry = {r['id']: r for r in read_json(Path(root) / 'campaigns/ni_drm/config/reactions.json')['reactions']}
    species = read_json(Path(root) / 'campaigns/ni_drm/config/species.json')
    for rid in ids:
        if rid not in registry or not registry[rid].get('enabled'):
            raise ValueError(f'Unknown or disabled reaction: {rid}')
        row = registry[rid]
        if row.get('parent_species') not in species or not row.get('broken_bond_elements'):
            raise ValueError(f'Reaction lacks executable dissociation specification: {rid}')
    # Association paper labels remain exact IDs; computational direction is explicit.
    rows = [{**registry[rid], 'execution_equation': registry[rid].get('computational_direction', registry[rid]['equation'])} for rid in ids]
    return {'name': doc.get('name', path.stem) if isinstance(doc, dict) else path.stem,
            'reactions': ids, 'rows': rows, 'source_path': str(path.resolve()), 'source_sha256': sha256(path)}


def environment():
    return {**os.environ, 'PATH': str(PYTHON.parent) + os.pathsep + os.environ.get('PATH', ''),
            'HF_HUB_CACHE': str(HF_CACHE), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}


def output_path(value, allowed=RUN_ROOT):
    allowed = Path(allowed).absolute()
    out = Path(value).absolute()
    if allowed.resolve() != allowed or out.resolve() != out:
        raise ValueError('Output paths must be physical paths, not symlinks/junctions')
    if out == allowed or not out.is_relative_to(allowed):
        raise ValueError(f'Output must be below {allowed}')
    return out


EXTERNAL_CALLS_DEFINITION = ('baseline reactant placements and relaxations (endpoint_phase_costs.json reactant.calls), '
                             'charged once per reaction and seed; the baseline product and frame preparation is not imported')


def external_reactants(root, cohort_doc, seed):
    """Same-pool comparison: pin the baseline seed run whose relaxed reactant records replace wf2.

    CHAVIS replicate k uses baseline seed k. Each cohort reaction needs a complete baseline endpoints stage; the
    checkpoint and its artifact hashes enter the request fingerprint, so a changed source refuses to resume.
    On a machine without the records (Windows preflight) the manifest is returned with available=False."""
    root = Path(root)
    seed_dir = root / f'seed{seed}'
    doc = {'kind': 'enumeration_baseline_reactant_records', 'root': str(root), 'seed_dir': str(seed_dir),
           'baseline_seed': seed, 'available': seed_dir.is_dir(), 'pins': None,
           'calls_definition': EXTERNAL_CALLS_DEFINITION,
           'preparation_scope': 'per reaction and seed; no species sharing between reactions'}
    if not doc['available']:
        return doc
    pins = {}
    for rid in cohort_doc['reactions']:
        checkpoint = seed_dir / 'reactions' / rid / 'checkpoint.json'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'baseline checkpoint missing for {rid}: {checkpoint}')
        entry = (read_json(checkpoint).get('completed_stages') or {}).get('endpoints')
        if not entry or entry.get('result', {}).get('status') != 'complete':
            raise ValueError(f'baseline endpoints stage is not complete for {rid}: {checkpoint}')
        reactant = (entry['result'].get('phase_costs') or {}).get('reactant') or {}
        pins[rid] = {'checkpoint_sha256': sha256(checkpoint), 'artifact_root': entry['result']['artifact_root'],
                     'artifact_hashes_sha256': digest_json(entry['artifact_hashes']),
                     'reactant_calls': reactant.get('calls'), 'reactant_placements': reactant.get('placements')}
    doc['pins'] = pins
    return doc


def request(cohort_doc, settings, seed=1, root=ROOT, reactant_records=None, surface='Ni:fcc111',
            search_scope=None):
    from ultra_fast_v1.planning import load_settings
    from .surfaces import DEFAULT_SURFACE, inspect_cache, surface_definition
    if type(seed) is not int or seed not in (1, 2, 3):
        raise ValueError('Execution replicate must be 1, 2 or 3; embedding remains frozen at 61453')
    cfg = load_settings(settings)
    if search_scope is not None:
        if search_scope not in ('first_success', 'all_pairs'):
            raise ValueError('search_scope must be first_success or all_pairs')
        cfg['search_scope'] = search_scope
    definition = surface_definition(surface)
    if surface != DEFAULT_SURFACE and cfg['adsorption_azimuth_step_deg'] != 'auto':
        raise ValueError('Additional surfaces require adsorption_azimuth_step_deg=auto (120 degrees is forbidden)')
    if reactant_records and surface != DEFAULT_SURFACE:
        raise ValueError('External reactant records are Ni:fcc111 only; other surfaces require native orientation search')
    req = {'schema': 'chavis_general_v2_2_20261019', 'cohort': cohort_doc,
           'settings': cfg, 'settings_source_sha256': sha256(settings),
           'params': read_json(Path(root) / 'reproduce/inputs/reference/params.json'),
           'seed': seed, 'seed_semantics': 'execution replicate label; frozen RDKit embedding seed 61453',
           'build_sha256': code_fingerprint(root), 'model_revision': REVISION,
           'registry_sha256': sha256(Path(root) / 'campaigns/ni_drm/config/reactions.json'),
           'species_sha256': sha256(Path(root) / 'campaigns/ni_drm/config/species.json'),
           'surface': definition}
    if surface != DEFAULT_SURFACE:
        req['surface_cache'] = inspect_cache(Path(root) / 'slab_store', surface)
    if reactant_records:
        req['schema'] = 'chavis_general_v2_2_same_pool_20261020'
        req['external_reactants'] = external_reactants(reactant_records, cohort_doc, seed)
    return req
