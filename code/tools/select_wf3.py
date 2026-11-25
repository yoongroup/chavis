import argparse
import json
import os
import sys
from pathlib import Path
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_k, '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'support' / 'runtime_support')]
from chavis.inputs import minima as M
from chavis.surface import slab_store
RIDS = M.enabled_reactions()


def minima_for(sdir, record, rebuild=False):
    path = sdir / 'minima.json'
    if path.is_file() and (not rebuild) and (path.stat().st_mtime >= (sdir / 'candidates.jsonl').stat().st_mtime):
        return json.loads(path.read_text())['minima']
    return [M.public(m) for m in M.build_for_species(sdir, record)]


def run_selection(run_dir, params, rebuild=False):
    run_dir = Path(run_dir)
    reactions = {r['id']: r for r in json.loads((ROOT / 'campaigns/ni_drm/config/reactions.json').read_text())['reactions']}
    out = {'params': params, 'surfaces': {}}
    for surf in sorted((p for p in run_dir.iterdir() if (p / 'slab_record_used.json').is_file())):
        name = json.loads((surf / 'slab_record_used.json').read_text())['name']
        record = slab_store.record_for_run(surf, ROOT / 'slab_store')
        rows = {}
        for rid in RIDS:
            sp = reactions[rid]['parent_species']
            sdir = surf / sp
            if not (sdir / 'candidates.jsonl').is_file():
                continue
            mins = minima_for(sdir, record, rebuild)
            requested = M.outcomes_for_reaction(reactions[rid])
            state = M.state_decision(
                mins,
                params.get('state_window_eV', M.DEFAULT_PARAMS['state_window_eV'])
            )
            outs = tuple(sorted(set(requested) | {'bound', 'floating'})) if state['both'] else requested
            chosen, report = M.select_with_fallback(mins, params, outs)
            rows[rid] = {
                'equation': reactions[rid]['equation'],
                'species': sp,
                'outcomes': list(outs),
                'requested_outcomes': list(requested),
                'state_class_db': reactions[rid].get('state_class'),
                'state': state,
                'fallback': report['fallback'],
                'n_minima': len(mins),
                'n_selected': len(chosen),
                'groups': report['groups'],
                'selected': [{k: m[k] for k in (
                    'form',
                    'energy_eV',
                    'delta_group_eV',
                    'delta_global_eV',
                    'representative',
                    'n_members',
                    'rotamer_of'
                )} for m in chosen]
            }
        out['surfaces'][surf.name] = rows
    (run_dir / 'wf3_selection.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--params', default=str(ROOT / 'config' / 'wf3_selection.json'))
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    res = run_selection(a.run_dir, M.load_params(a.params), a.rebuild)
    for s, rows in res['surfaces'].items():
        for rid, r in rows.items():
            print(f"{s:10s} {rid} {r['equation']:22s} minima {r['n_minima']:4d} -> selected {r['n_selected']:3d}")
