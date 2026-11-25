import os
for _k in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS'
):
    os.environ.setdefault(_k, '2')
import argparse
import collections
from copy import deepcopy
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'support' / 'runtime_support')]
from chavis.inputs import fs_build as F
from chavis.inputs import minima as M
from chavis.surface import slab_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--params', default=str(ROOT / 'config' / 'wf3_selection.json'))
    ap.add_argument('--surface', default=None)
    ap.add_argument('--reaction', default=None)
    ap.add_argument(
        '--max-is',
        type=int,
        default=None,
        help='smoke tests: only the lowest N IS per reaction'
    )
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    p = F.clean_params(M.load_params(a.params))
    log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)
    reactions = {r['id']: r for r in json.loads((ROOT / 'campaigns/ni_drm/config/reactions.json').read_text())['reactions']}
    wf3 = json.loads((run_dir / 'wf3_is.json').read_text())
    summary_path = run_dir / 'wf4_fs.json'
    out = json.loads(summary_path.read_text()) if summary_path.is_file() else {'surfaces': {}}
    for saved_rxns in out['surfaces'].values():
        for saved in saved_rxns.values():
            saved.setdefault(
                'provenance',
                {
                    'status': 'legacy_unknown',
                    'params': None,
                    'relax': None,
                    'selection': None
                }
            )
    out.update(params=p, relax=F.FS_RELAX, params_relax_scope='last_invocation')
    selection = {'surface': a.surface, 'reaction': a.reaction, 'max_is': a.max_is}
    for surf, rxns in wf3['surfaces'].items():
        if a.surface and surf != a.surface:
            continue
        record = slab_store.record_for_run(run_dir / surf, ROOT / 'slab_store')
        by_species = collections.defaultdict(list)
        for rid, r in rxns.items():
            if not a.reaction or rid == a.reaction:
                by_species[r['species']].append(rid)
        for species, rids in by_species.items():
            sp = F.Species(run_dir / surf / species, record)
            refined = [json.loads(l) for l in (sp.dir / 'refined.jsonl').open()]
            by_rep = {r['minimum_representative']: r for r in refined}
            wf2_min = json.loads((sp.dir / 'minima.json').read_text())['minima']
            wf2_rows = {r['index']: r for r in (json.loads(l) for l in (sp.dir / 'candidates.jsonl').open())}
            for rid in rids:
                reaction = reactions[rid]
                is_rows = [by_rep[x['minimum_representative']] for x in rxns[rid]['is']]
                if a.max_is:
                    is_rows = is_rows[:a.max_is]
                log(f"[{surf}] {rid} {reaction['equation']}: {len(is_rows)} IS")
                rows = F.fs_for_reaction(
                    sp,
                    reaction,
                    is_rows,
                    refined,
                    wf2_min,
                    wf2_rows,
                    p,
                    log
                )
                with (sp.dir / f'fs_{rid}.jsonl').open('w') as fh:
                    for r in rows:
                        fh.write(json.dumps({'reaction': rid, **r}) + '\n')
                src = collections.Counter((r['source'] for r in rows))
                ok = collections.Counter((r['source'] for r in rows if r['product_match']))
                rep = collections.Counter(('+'.join(r['group_sources']) for r in rows if r.get('fs_representative')))
                out['surfaces'].setdefault(surf, {})[rid] = {
                    'equation': reaction['equation'],
                    'species': species,
                    'n_is': len(is_rows),
                    'relaxed_or_collected': dict(src),
                    'product_match': dict(ok),
                    'distinct_fs': dict(rep),
                    'outcomes': dict(collections.Counter((r['outcome'] for r in rows))),
                    'vertical_bond_fallback': sum((bool(r.get('vertical_bond_fallback')) for r in rows)),
                    'provenance': {
                        'status': 'recorded',
                        'params': deepcopy(p),
                        'relax': deepcopy(F.FS_RELAX),
                        'params_path': str(Path(a.params).resolve()),
                        'selection': selection.copy(),
                        'available_is_count': len(rxns[rid]['is']),
                        'selected_is_representatives': [r['minimum_representative'] for r in is_rows]
                    }
                }
                log(f'[{surf}] {rid}: rows {dict(src)}, product match {dict(ok)}, distinct {dict(rep)}')
                summary_path.write_text(json.dumps(out, indent=1))
            sp.close()
    log('WF4 DONE')


if __name__ == '__main__':
    main()
