import argparse
import collections
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'support' / 'runtime_support')]
from chavis.inputs import fs_build as F
from chavis.inputs import pairs as P
from chavis.inputs import minima as M
from chavis.surface import slab_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--surface', default=None)
    ap.add_argument('--reaction', default=None)
    ap.add_argument(
        '--params',
        default=None,
        help='override the saved wf3 intercalation-filter choice with this selection JSON'
    )
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)
    reactions = {r['id']: r for r in json.loads((ROOT / 'campaigns/ni_drm/config/reactions.json').read_text())['reactions']}
    wf3 = json.loads((run_dir / 'wf3_is.json').read_text())
    if a.params is not None:
        exclude_intercalated = M.load_params(a.params)['exclude_intercalated']
        exclude_intercalated_source = str(a.params)
    elif 'exclude_intercalated' in (wf3.get('params') or {}):
        exclude_intercalated = wf3['params']['exclude_intercalated']
        exclude_intercalated_source = 'wf3_is.json.params'
    else:
        selection_path = ROOT / 'config' / 'wf3_selection.json'
        exclude_intercalated = M.load_params(selection_path)['exclude_intercalated']
        exclude_intercalated_source = str(selection_path)
    prev = json.loads((run_dir / 'wf5_pairs.json').read_text()) if (run_dir / 'wf5_pairs.json').is_file() else {}
    out = {
        'rule': 'generated FS -> the IS it was derived from; spontaneous FS -> IS derived from it by shortening the broken bond, replaced by an existing IS when symmetry-equivalent; pairs merged when one op + one permutation maps IS and FS',
        'tolerances_A': {'ads_rmsd': P.SAME_ADS_A, 'slab_max': P.SAME_SLAB_A},
        'exclude_intercalated': exclude_intercalated,
        'exclude_intercalated_source': exclude_intercalated_source,
        'surfaces': prev.get('surfaces', {})
    }
    for surf, rxns in wf3['surfaces'].items():
        if a.surface and surf != a.surface:
            continue
        record = slab_store.record_for_run(run_dir / surf, ROOT / 'slab_store')
        for rid, r in rxns.items():
            if a.reaction and rid != a.reaction:
                continue
            sp = F.Species(run_dir / surf / r['species'], record)
            fs_path = sp.dir / f'fs_{rid}.jsonl'
            if not fs_path.is_file():
                log(f'[{surf}] {rid}: no wf4 FS file, skipped')
                continue
            refined = {x['minimum_representative']: x for x in (json.loads(l) for l in (sp.dir / 'refined.jsonl').open())}
            is_rows = [refined[x['minimum_representative']] for x in r['is']]
            fs_rows = [json.loads(l) for l in fs_path.open()]
            pairs, derived = P.pair_reaction(
                sp,
                reactions[rid],
                is_rows,
                fs_rows,
                log,
                exclude_intercalated=exclude_intercalated
            )
            with (sp.dir / f'pairs_{rid}.jsonl').open('w') as fh:
                for index, p in enumerate(pairs):
                    p['pair_id'] = f'{rid}_row_{index:06d}'
                    fh.write(json.dumps(p) + '\n')
            with (sp.dir / f'is_from_fs_{rid}.jsonl').open('w') as fh:
                for d in derived:
                    fh.write(json.dumps({'reaction': rid, **d}) + '\n')
            reps = [p for p in pairs if p['pair_representative']]
            out['surfaces'].setdefault(surf, {})[rid] = {
                'equation': reactions[rid]['equation'],
                'route': P.ROUTES.get(reactions[rid].get('default_route')),
                'n_is': len(is_rows),
                'n_fs_matching': sum((1 for f in fs_rows if f['product_match'] and f['converged'])),
                'n_pairs': len(pairs),
                'n_distinct_pairs': len(reps),
                'distinct_by_source': dict(collections.Counter(('+'.join(p['pair_group_sources']) for p in reps))),
                'by_pairing': dict(collections.Counter((p['pairing'] for p in pairs))),
                'derived_is': dict(collections.Counter((d.get('result') for d in derived))),
                'is_without_pair': sorted({x['minimum_representative'] for x in is_rows} - {p['paired_is'] for p in pairs}),
                'is_far_above_lowest': sorted(
                    {p['paired_is'] for p in pairs if p.get('is_far_above_lowest')},
                    key=str
                )
            }
            log(f"[{surf}] {rid} {reactions[rid]['equation']}: {out['surfaces'][surf][rid]}")
            sp.close()
    (run_dir / 'wf5_pairs.json').write_text(json.dumps(out, indent=1))
    log('WF5 PAIRS DONE')


if __name__ == '__main__':
    main()
