"""Single-reaction ultra-fast entry point. Importing this module never loads UMA."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid



def _sha256_file(path):
    """SHA256 with bounded memory; identical to hashing the complete file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

ROOT = Path(__file__).resolve().parents[1]
from generic_v2.common import RUN_ROOT, output_path


def finite_energy(value, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'nonfinite/missing reactant energy: {source}')
    return float(value)


def load_reference_energy(run_dir, surface, rid, species_dir, all_pairs):
    """Selected refined pool INCLUDING reactants without pairs, plus derived pair reactants."""
    wf3 = json.loads((Path(run_dir) / 'wf3_is.json').read_text(encoding='utf-8'))
    selected = wf3['surfaces'][surface][rid]['is']
    refined = {str(r['minimum_representative']): r for r in
               (json.loads(l) for l in (Path(species_dir) / 'refined.jsonl').read_text(encoding='utf-8').splitlines() if l.strip())}
    energies = []
    for row in selected:
        key = str(row['minimum_representative'])
        if key not in refined:
            raise ValueError(f'selected reactant missing from refined.jsonl: {key}')
        energies.append(finite_energy(refined[key].get('energy_eV'), f'refined:{key}'))
    energies += [finite_energy(p.get('is_energy_eV'), f"pair:{p.get('pair_group')}") for p in all_pairs]
    if not energies:
        raise ValueError('empty reactant reference pool')
    return min(energies)


def request_fingerprint(inputs, settings, params):
    files = []
    for name in ('src', 'tools', 'config', 'campaigns/ni_drm/config',
                 'support/runtime_support/automation_search', 'ultra_fast_v1'):
        files += [p for p in (ROOT / name).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.log')]
    payload = {'protocol': 'ultra_fast_v2_2_20261019', 'settings': settings, 'params': params,
               'code': {p.relative_to(ROOT).as_posix(): _sha256_file(p) for p in sorted(files)},
               'inputs': {str(p): _sha256_file(p) for p in sorted(set(inputs))}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', type=Path, help=f'Physical run directory below {RUN_ROOT}')
    ap.add_argument('--params', type=Path, required=True)
    ap.add_argument('--settings', type=Path, required=True)
    ap.add_argument('--surface', required=True)
    ap.add_argument('--reaction', required=True)
    ap.add_argument('--cost-file', type=Path, required=True, help=f'Fresh cost file below {RUN_ROOT}')
    args = ap.parse_args(argv)
    if os.name == 'posix':
        for target in (args.run_dir, args.cost_file):
            output_path(target, allowed=RUN_ROOT)
    if args.cost_file.exists():
        raise FileExistsError(args.cost_file)
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'support/runtime_support')]
    from .planning import load_settings, load_form_metadata
    from .cost import SegmentLedger
    from .scheduler import run_reaction, write_json
    from chavis.inputs.fs_build import Species
    from chavis.surface import slab_store
    from chavis.surface.metal_mode import parse
    from chavis.runtime.run_snapshot import start_run, end_run, DEFAULT_CODE

    started = time.monotonic()
    species, ledger, attempt, out = None, None, None, None
    settings, result = None, None
    output_owned = False
    error = None
    status = 'complete'
    def calls():
        return int(species.factory.calls) if species is not None and species.factory is not None else 0
    try:
        settings = load_settings(args.settings)
        params = json.loads(args.params.read_text(encoding='utf-8'))
        if params.get('ts_route_override') != 'scan_then_direct':
            raise ValueError('ultra v2 requires the frozen scan_then_direct base parameters')
        if parse(params['refine_fix_layers']) != parse(params['ts_fix_layers']):
            raise ValueError('ultra v2 reference energies require identical refine and TS fixed layers')
        run = args.run_dir.resolve()
        wf3 = json.loads((run / 'wf3_is.json').read_text(encoding='utf-8'))
        rx = wf3['surfaces'][args.surface][args.reaction]
        species_dir = run / args.surface / rx['species']
        pair_path = species_dir / f'pairs_{args.reaction}.jsonl'
        all_pairs = [json.loads(l) for l in pair_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        if any(parse(p['fixed_layers']) != parse(params['ts_fix_layers']) for p in all_pairs):
            raise ValueError('pair fixed layers differ from the fixed full-pool energy reference')
        pairs = [p for p in all_pairs if p['pair_representative']]
        reference = load_reference_energy(run, args.surface, args.reaction, species_dir, all_pairs)
        metadata = load_form_metadata(species_dir, args.reaction)
        reaction_path = ROOT / 'campaigns/ni_drm/config/reactions.json'
        reactions = {r['id']: r for r in json.loads(reaction_path.read_text(encoding='utf-8'))['reactions']}
        reaction = reactions[args.reaction]
        out = species_dir / f'ts_ultra_{args.reaction}'
        if out.exists():
            raise FileExistsError(f'ultra v2 has no disk continuation; use a fresh output: {out}')
        out.mkdir()
        output_owned = True
        record = slab_store.record_for_run(run / args.surface, ROOT / 'slab_store')
        inputs = [args.params, args.settings, run / 'wf3_is.json', pair_path,
                  reaction_path, species_dir / 'refined.jsonl', species_dir / 'gas/molecule.json',
                  run / args.surface / 'slab_record_used.json']
        derived = species_dir / f'is_from_fs_{args.reaction}.jsonl'
        if derived.is_file():
            inputs.append(derived)
        minima = species_dir / 'minima.json'
        if minima.is_file():
            inputs.append(minima)
        used = json.loads((run / args.surface / 'slab_record_used.json').read_text(encoding='utf-8'))
        store = Path(used.get('store') or ROOT / 'slab_store')
        inputs.append(store / used['name'] / 'record.json')
        fingerprint = request_fingerprint(inputs, settings, params)
        attempt = run / 'ts_attempts' / ('ultra_' + time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex)
        from automation_search.uma_model import ASSETS, REVISION
        start_run(attempt, ROOT, [sys.executable, '-B', '-m', 'ultra_fast_v1.cli', *(argv if argv is not None else sys.argv[1:])],
                  config={'base': params, 'ultra_fast': settings}, input_paths=inputs,
                  model={'name': 'uma-s-1p2p1', 'task': 'oc20', 'revision': REVISION,
                         'expected_assets': ASSETS, 'checkpoint_note': 'verified factory assets saved in model.json'},
                  workers={'processes': 1, 'in_process_interleaving': True},
                  code_entries=(*DEFAULT_CODE, 'ultra_fast_v1', 'reproduce/drivers/ultra_fast.py',
                                'reproduce/inputs/reference', 'reproduce/inputs/ultra_fast'))
        species = Species(species_dir, record)
        ledger = SegmentLedger(calls, out / 'segments.jsonl')
        with ledger.segment('model_initialization'):
            species.ensure_calc()
        result = run_reaction(species, pairs, reaction, params, settings, metadata, out, ledger,
                              reference, attempt.name, fingerprint,
                              log=lambda s: print(time.strftime('%H:%M:%S'), s, flush=True))
        write_json(attempt / 'model.json', {'assets': species.factory.assets,
                   'model_load_seconds': species.factory.load_seconds,
                   'inference': species.factory.metadata(),
                   'scope': 'one species model for this reaction', 'slab_record': record,
                   'species': rx['species']})
        write_json(out / 'cost_ledger.json', ledger.summary())
        if result['model_calls'] != calls():
            raise RuntimeError('reaction total differs from raw factory counter')
        end_run(attempt, 'complete')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        if output_owned:
            write_json(out / 'error.json', {'error': error, 'calls': calls(),
                       'seconds': time.monotonic() - started,
                       'status': status, 'search_scope': settings['search_scope'],
                       'continuation': 'reaction_boundary_fresh_attempt_only',
                       'candidate_error': getattr(exc, 'chavis_candidate', None)})
        if attempt is not None and (attempt / '_snapshot/run.json').exists():
            end_run(attempt, 'failed', error=error)
        print(error, file=sys.stderr, flush=True)
    finally:
        measured_calls = calls()
        models_created = int(species is not None and species.factory is not None)
        if species is not None:
            species.close()
        # Never overwrite even a failed prior attempt's cost envelope.
        args.cost_file.parent.mkdir(parents=True, exist_ok=True)
        with args.cost_file.open('x', encoding='utf-8') as f:
            json.dump({'tool': 'ultra_fast_v1.cli', 'calls': measured_calls,
                       'seconds': time.monotonic() - started,
                       'models_created': models_created,
                       'status': status,
                       'search_scope': settings['search_scope'] if settings else None,
                       'error': error}, f, indent=2, allow_nan=False)
            f.write('\n')
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
