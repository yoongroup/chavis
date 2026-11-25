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
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / 'src'),
    str(ROOT / 'support' / 'runtime_support'),
    str(ROOT / 'tools')
]
from chavis.inputs import minima as M
from chavis.surface import slab_store
from chavis.inputs import is_refine as R
import select_wf3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--params', default=str(ROOT / 'config' / 'wf3_selection.json'))
    ap.add_argument('--surface', default=None)
    ap.add_argument('--species', default=None)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    params = M.load_params(a.params)
    log = lambda s: print(time.strftime('%H:%M:%S'), s, flush=True)
    sel = select_wf3.run_selection(run_dir, params)
    out = {'params': params, 'surfaces': {}}
    for surf, rows in sel['surfaces'].items():
        if a.surface and surf != a.surface:
            continue
        name = json.loads((run_dir / surf / 'slab_record_used.json').read_text())['name']
        record = slab_store.record_for_run(run_dir / surf, ROOT / 'slab_store')
        per_species = collections.defaultdict(dict)
        for rid, r in rows.items():
            for m in r['selected']:
                per_species[r['species']][m['representative']] = m
        refined = {}
        for sp, ms in per_species.items():
            if a.species and sp != a.species:
                continue
            mins = json.loads((run_dir / surf / sp / 'minima.json').read_text())['minima']
            full = {m['representative']: m for m in mins}
            todo = [full[k] for k in sorted(ms)]
            log(f"[{surf}/{sp}] re-relaxing {len(todo)} minima, fixed layers {params['refine_fix_layers']}")
            refined[sp] = R.refine_species(
                run_dir / surf / sp,
                record,
                todo,
                params['refine_fix_layers'],
                log
            )
        surf_out = {}
        for rid, r in rows.items():
            if r['species'] not in refined:
                continue
            outs = [r['fallback']['used']] if r.get('fallback') else r['outcomes']
            chosen_reps = {m['representative'] for m in r['selected']}
            mine = [x for x in refined[r['species']] if x['minimum_representative'] in chosen_reps]
            iss = R.is_candidates(
                mine,
                outs,
                params['is_per_reaction'],
                params['is_require_binder_bonded'],
                params['is_binder_ratio_scale'],
                params['is_binder_min_above_eV'],
                params['exclude_intercalated']
            )
            surf_out[rid] = {
                'equation': r['equation'],
                'species': r['species'],
                'outcomes': outs,
                'n_handed_over': len(mine),
                'n_after_merge': sum((1 for x in mine if x.get('is_representative'))),
                'n_unconverged': sum((1 for x in mine if not x.get('converged', True))),
                'n_intercalated': sum((R.intercalated(x) for x in mine)),
                'exclude_intercalated': params['exclude_intercalated'],
                'n_upside_down': len(R.upside_down(
                    mine,
                    params['is_binder_ratio_scale'],
                    params['is_binder_min_above_eV']
                )),
                'require_binder_bonded': params['is_require_binder_bonded'],
                'n_is': len(iss),
                'transitions': dict(collections.Counter((x['transition']['outcome'] for x in mine if x['transition']))),
                'form_changed': sum((bool(x['transition'] and x['transition']['form_changed']) for x in mine)),
                'is': [{
                    'minimum_representative': x['minimum_representative'],
                    'energy_eV': x['energy_eV'],
                    'form_before_after': x['transition']['form']
                } for x in iss]
            }
            log(f"[{surf}] {rid} {r['equation']}: handed over {len(mine)} -> after merge {surf_out[rid]['n_after_merge']} -> IS {len(iss)}; {surf_out[rid]['transitions']}")
        out['surfaces'][surf] = surf_out
    (run_dir / 'wf3_is.json').write_text(json.dumps(out, indent=1))
    log('WF3 DONE')


if __name__ == '__main__':
    main()
