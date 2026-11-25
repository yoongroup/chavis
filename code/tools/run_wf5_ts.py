import os
for _k in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS'
):
    os.environ.setdefault(_k, '2')
import argparse
import json
import hashlib
from importlib.metadata import version
import sys
import time
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'support' / 'runtime_support')]
from chavis.inputs import fs_build as F
from chavis.inputs import minima as M
from chavis.inputs.adsorbate_search import gpu_guard
from chavis.search import route_v2 as R
from chavis.surface import slab_store
from chavis.runtime.ts_cost import build_ledger
from chavis.runtime.hashing import sha256_file
from chavis.runtime.run_snapshot import start_run, end_run


def request_fingerprint(record, pair, reaction, params, species_dir):
    files = list((ROOT / 'src').rglob('*.py'))
    files += list((ROOT / 'support/runtime_support/automation_search').rglob('*.py'))
    files.append(Path(__file__))
    code = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)}
    gas = Path(species_dir) / 'gas/molecule.json'
    from automation_search.uma_model import ASSETS, REVISION
    payload = {
        'schema': 2,
        'pair': pair,
        'reaction': reaction,
        'slab_structure': record.get('structure'),
        'model': record.get('model'),
        'params': params,
        'settings': R.TS_SETTINGS,
        'code': code,
        'model_assets_required': ASSETS,
        'model_revision': REVISION,
        'candidate_order': (('scan_stage_a_peak_outward>scan_stage_b>direct scan_then_direct' if params.get('ts_route_override') == 'scan_then_direct' else R.CANDIDATE_ORDER) if params.get('mode') else 'legacy'),
        'fast_stop_share': R.FAST_STOP_SHARE if params.get('mode') else None,
        'gas': json.loads(gas.read_text()) if gas.exists() else None,
        'versions': {name: version(name) for name in [
            'ase',
            'numpy',
            'scipy',
            'torch',
            'fairchem.core'
        ]}
    }
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':')
    ).encode()).hexdigest()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--params', default=str(ROOT / 'config' / 'wf3_selection.json'))
    ap.add_argument('--surface', default=None)
    ap.add_argument('--reaction', default=None)
    ap.add_argument(
        '--max-pairs',
        type=int,
        default=None,
        help='smoke tests: first N pairs per reaction, ordered by IS energy then reaction energy'
    )
    ap.add_argument('--shard', default='0/1')
    ap.add_argument(
        '--light',
        action='store_true',
        help='quick trial: pairs ordered by IS energy (use --max-pairs), stop at the first reaction TS of either label, no scan stage B; results in ts_light_<reaction>'
    )
    ap.add_argument('--mode', choices=('fast', 'slow'), default=None)
    ap.add_argument('--dimer-max-steps', type=int, default=None)
    ap.add_argument(
        '--all-pairs',
        action='store_true',
        help='collection only: run every pair row with a unique row index'
    )
    return ap


def output_name(rid, mode=None, light=False):
    R.check_mode(mode, light)
    return f'ts_{mode}_{rid}' if mode else f'ts_light_{rid}' if light else f'ts_{rid}'


def select_pairs(rows, all_pairs=False):
    selected = [
        {
            **p,
            'collection_pair_index': p.get('collection_pair_index', i),
            'pair_id': p.get('pair_id', f'row_{i:06d}')
        } if all_pairs else p for i,
        p in enumerate(rows) if all_pairs or p['pair_representative']
    ]
    if all_pairs and (any((type(p['collection_pair_index']) is not int or p['collection_pair_index'] < 0 for p in selected)) or len({p['collection_pair_index'] for p in selected}) != len(selected)):
        raise ValueError('Invalid or duplicate collection_pair_index')
    if all_pairs and len({p['pair_id'] for p in selected}) != len(selected):
        raise ValueError('Duplicate collection pair_id')
    if all_pairs and any((not isinstance(p['pair_id'], str) or not p['pair_id'] or any((c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in p['pair_id'])) for p in selected)):
        raise ValueError('Invalid collection pair_id')
    return sorted(selected, key=lambda p: (p['is_energy_eV'], p['reaction_energy_eV']))


def pair_output_name(pair):
    if 'collection_pair_index' in pair:
        return f"pair_{pair['pair_id']}"
    return f"pair_{pair['pair_group']:03d}"


def main():
    ap = parser()
    a = ap.parse_args()
    R.check_mode(a.mode, a.light)
    run_dir = Path(a.run_dir).resolve()
    raw = M.load_params(a.params)
    if a.dimer_max_steps is not None:
        raw = {**raw, 'dimer_max_steps': a.dimer_max_steps}
    params = {**R.clean_ts_params(raw), 'light': a.light, 'mode': a.mode}
    if a.all_pairs:
        params['collection_all_pairs'] = True
    shard, nshard = map(int, a.shard.split('/'))
    if nshard < 1 or not 0 <= shard < nshard or (a.max_pairs is not None and a.max_pairs < 1):
        ap.error('require 0 <= shard < nshard and positive --max-pairs')
    attempt = run_dir / 'ts_attempts' / (time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex)
    inputs = [
        Path(a.params),
        run_dir / 'wf3_is.json',
        ROOT / 'campaigns/ni_drm/config/reactions.json'
    ]
    for used_path in sorted(run_dir.glob('*/slab_record_used.json')):
        inputs.append(used_path)
        used = json.loads(used_path.read_text(encoding='utf-8'))
        inputs.append(Path(used.get('store') or ROOT / 'slab_store') / used['name'] / 'record.json')
    inputs += sorted(run_dir.glob('*/*/pairs_*.jsonl'))
    inputs += sorted(run_dir.glob('*/*/gas/molecule.json'))
    from automation_search.uma_model import ASSETS, REVISION
    start_run(
        attempt,
        ROOT,
        [sys.executable, *sys.argv],
        config=params,
        input_paths=inputs,
        model={
            'name': 'uma-s-1p2p1',
            'task': 'oc20',
            'revision': REVISION,
            'expected_assets': ASSETS,
            'checkpoint_note': 'actual verified assets recorded per species after model creation'
        },
        workers={'shard': shard, 'nshard': nshard},
        extra_code=[Path(__file__)]
    )
    try:
        failed = execute(a, run_dir, params, shard, nshard, attempt)
    except BaseException as exc:
        end_run(attempt, 'failed', error=repr(exc))
        raise
    end_run(attempt, 'completed_with_errors' if failed else 'complete')


def execute(a, run_dir, params, shard, nshard, attempt):
    failed = 0
    log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)
    reactions = {r['id']: r for r in json.loads((ROOT / 'campaigns/ni_drm/config/reactions.json').read_text())['reactions']}
    wf3 = json.loads((run_dir / 'wf3_is.json').read_text())
    k = -1
    for surf, rxns in wf3['surfaces'].items():
        if a.surface and surf != a.surface:
            continue
        record = slab_store.record_for_run(run_dir / surf, ROOT / 'slab_store')
        for rid, r in rxns.items():
            if a.reaction and rid != a.reaction:
                continue
            path = run_dir / surf / r['species'] / f'pairs_{rid}.jsonl'
            if not path.is_file():
                continue
            pairs = select_pairs(list(map(json.loads, path.open())), a.all_pairs)
            if a.max_pairs:
                pairs = pairs[:a.max_pairs]
            sp = None
            for p in pairs:
                k += 1
                if k % nshard != shard:
                    continue
                out = run_dir / surf / r['species'] / output_name(rid, a.mode, a.light) / pair_output_name(p)
                fingerprint = request_fingerprint(
                    record,
                    p,
                    reactions[rid],
                    params,
                    run_dir / surf / r['species']
                )
                if (out / 'result.json').is_file():
                    saved = json.loads((out / 'result.json').read_text())
                    if saved.get('request_fingerprint') != fingerprint:
                        raise ValueError(f'Existing TS result has missing or different request fingerprint: {out}. Preserve this result and use a new run directory for changed inputs/settings.')
                    if a.mode:
                        expected = build_ledger(saved, saved['attempt_id'])
                        ledger_path = out / 'cost_ledger.json'
                        if not ledger_path.is_file() or json.loads(ledger_path.read_text()) != expected:
                            raise ValueError(f'Missing or inconsistent TS cost ledger: {out}')
                    continue
                if sp is None:
                    sp = F.Species(run_dir / surf / r['species'], record)
                gpu_guard(log=log)
                log(f"[{surf}] {rid} {reactions[rid]['equation']} pair {p['pair_group']} ({p['route']}, IS {p['paired_is']}, FS {p['fs_source']})")
                call_start = sp.factory.calls if sp.factory is not None else 0
                pair_start = time.monotonic()
                try:
                    res = R.run_pair(sp, p, reactions[rid], params, out, log)
                    if a.all_pairs:
                        res['collection_pair_index'] = p['collection_pair_index']
                        res['pair_id'] = p['pair_id']
                    res['attempt_id'] = attempt.name
                    if a.mode:
                        ledger = build_ledger(res, attempt.name)
                        (out / 'cost_ledger.json').write_bytes(json.dumps(ledger, indent=1).encode('utf-8'))
                    if sp.factory is not None:
                        model_rec = {
                            'assets': sp.factory.assets,
                            'model_load_seconds': sp.factory.load_seconds,
                            'scope': 'one species model, not additive per pair',
                            'slab_record': record,
                            'species': r['species']
                        }
                        (attempt / f'model_{surf}_{rid}.json').write_bytes(json.dumps(model_rec, indent=1).encode('utf-8'))
                    res['request_fingerprint'] = fingerprint
                    (out / 'result.json').write_bytes(json.dumps(res, indent=1).encode('utf-8'))
                    log(f"    -> target {res['target_found']}, barrier {res['barrier_forward_eV']}; other-endpoint reaction TS {res['reaction_ts_other_endpoints']} {res['barrier_other_endpoints_eV']}; {res['seconds']} s")
                except Exception as exc:
                    failed += 1
                    out.mkdir(parents=True, exist_ok=True)
                    call_end = sp.factory.calls if sp.factory is not None else None
                    (out / 'error.json').write_bytes(json.dumps({
                        'error': f'{type(exc).__name__}: {exc}',
                        'attempt_id': attempt.name,
                        'calls': None if call_end is None else call_end - call_start,
                        'seconds': time.monotonic() - pair_start,
                        'cost_scope': 'runner pair attempt including model initialization if needed',
                        'candidate_error': getattr(exc, 'chavis_candidate', None),
                        'stage_cost_status': 'see candidate_error payload, <seed>_candidate_error.json and candidates.jsonl'
                    }).encode('utf-8'))
                    log(f'    -> ERROR {type(exc).__name__}: {exc}')
            if sp is not None:
                sp.close()
    log('WF5 TS DONE')
    return failed


if __name__ == '__main__':
    main()
