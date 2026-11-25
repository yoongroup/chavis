from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
SAME_A = 0.1
ROTAMER_HEIGHT_TOL_A = 0.1
DEFAULT_PARAMS = {
    'count': 2,
    'window_eV': 0.3,
    'rule': 'both',
    'include_rotamers': True,
    'refine_fix_layers': '3,4',
    'is_per_reaction': 'all',
    'is_require_binder_bonded': True,
    'is_binder_ratio_scale': 1.3,
    'is_binder_min_above_eV': 1.0,
    'exclude_intercalated': True,
    'state_window_eV': 0.2
}
ROLE_TO_OUTCOME = {
    'surface_contact': 'bound',
    'weak_contact': 'bound',
    'detached_molecule': 'floating'
}


def _outcome(row):
    if row.get('outcome'):
        return row['outcome']
    lab = row['labels']
    return 'dissociated' if lab['dissociation']['flag'] else 'floating' if lab['desorption']['flag'] else 'bound'


def form(row):
    oc = _outcome(row)
    cs = row['labels']['binding']['contacts']
    base = '+'.join(sorted((f"{c['element']}@{c.get('site')}" for c in cs))) if cs else row['labels']['binding']['mode']
    return base if oc == 'bound' else f'{oc}:{base}'


def _heights(pos, symbols, order):
    return np.concatenate([np.sort(pos[
        [i for i, s in enumerate(symbols) if s == el],
        2
    ]) for el in order])


def distinct_minima(rows, symbols, ops, cell):
    from chavis.surface.symmetry import equivalent
    order = sorted(set(symbols))
    rows = sorted(
        (r for r in rows if r.get('energy_eV') is not None and r.get('converged', True)),
        key=lambda r: (r['energy_eV'], r['index'])
    )
    minima = []
    for r in rows:
        p = np.asarray(r['final_adsorbate_positions'], float)
        h = _heights(p, symbols, order)
        oc, fm = (_outcome(r), form(r))
        for m in minima:
            if m['outcome'] == oc and np.max(np.abs(h - m['_h'])) < 2 * SAME_A and equivalent(m['_p'], p, symbols, ops, cell, SAME_A):
                m['members'].append(r['index'])
                break
        else:
            minima.append({
                'outcome': oc,
                'form': fm,
                'energy_eV': r['energy_eV'],
                'representative': r['index'],
                'members': [r['index']],
                '_p': p,
                '_h': h
            })
    groups = {}
    for m in minima:
        groups.setdefault((m['outcome'], m['form']), []).append(m)
    for (oc, fm), ms in groups.items():
        g_min = ms[0]['energy_eV']
        fam = []
        for m in ms:
            m['delta_group_eV'] = round(m['energy_eV'] - g_min, 4)
            for f in fam:
                if np.max(np.abs(m['_h'] - f[0])) < ROTAMER_HEIGHT_TOL_A:
                    m['rotamer_of'] = f[1]
                    break
            else:
                m['rotamer_of'] = None
                fam.append((m['_h'], m['representative']))
    e0 = min((m['energy_eV'] for m in minima), default=None)
    for m in minima:
        m['delta_global_eV'] = round(m['energy_eV'] - e0, 4)
        m['n_members'] = len(m['members'])
    return minima


def select(minima, params=None, outcomes=('bound',)):
    p = {**DEFAULT_PARAMS, **(params or {})}
    if p['rule'] not in ('count', 'window', 'both', 'either'):
        raise ValueError('rule must be count, window, both or either')
    chosen, report = ([], [])
    groups = {}
    for m in minima:
        if m['outcome'] in outcomes:
            groups.setdefault(m['form'], []).append(m)
    for fm, ms in sorted(groups.items(), key=lambda kv: kv[1][0]['energy_eV']):
        pool = ms if p['include_rotamers'] else [m for m in ms if m['rotamer_of'] is None]
        picked = []
        for rank, m in enumerate(pool):
            in_count = rank < int(p['count'])
            in_window = m['delta_group_eV'] <= float(p['window_eV']) + 1e-12
            ok = {
                'count': in_count,
                'window': in_window,
                'both': in_count and in_window,
                'either': in_count or in_window
            }[p['rule']]
            if ok:
                picked.append(m)
        chosen += picked
        report.append({
            'form': fm,
            'n_minima': len(ms),
            'n_after_rotamer_rule': len(pool),
            'n_selected': len(picked),
            'lowest_delta_global_eV': ms[0]['delta_global_eV']
        })
    return (chosen, {'params': p, 'outcomes': list(outcomes), 'groups': report})


def public(m):
    return {k: v for k, v in m.items() if not k.startswith('_')}


def build_for_species(species_dir, record, params=None):
    from chavis.surface import slab_store
    from chavis.surface.symmetry import surface_operations
    species_dir = Path(species_dir)
    rows = [json.loads(l) for l in (species_dir / 'candidates.jsonl').open()]
    symbols = json.loads((species_dir / 'gas' / 'molecule.json').read_text())['symbols']
    slab = slab_store.to_atoms(record)
    minima = distinct_minima(rows, symbols, surface_operations(slab), slab.cell.array)
    (species_dir / 'minima.json').write_text(json.dumps(
        {
            'rule': 'symmetry-merged final structures, grouped by outcome and atom@site',
            'same_A': SAME_A,
            'rotamer_height_tol_A': ROTAMER_HEIGHT_TOL_A,
            'n_results': len(rows),
            'n_minima': len(minima),
            'minima': [public(m) for m in minima]
        },
        indent=1
    ))
    return minima


STATE_CLASS_OUTCOMES = {
    'adsorbed_to_adsorbed': ('bound',),
    'may_detach_to_adsorbed': ('bound', 'floating')
}


def outcomes_for_reaction(reaction):
    if isinstance(reaction, str):
        reaction = reactions_db()[reaction]
    return STATE_CLASS_OUTCOMES[reaction.get('state_class') or 'adsorbed_to_adsorbed']


def reactions_db(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[3] / 'campaigns' / 'ni_drm' / 'config' / 'reactions.json'
    return {r['id']: r for r in json.loads(path.read_text())['reactions']}


def enabled_reactions(path=None):
    enabled = [
        rid for rid,
        r in reactions_db(path).items() if r.get('enabled') and r.get('parent_species')
    ]
    requested = os.environ.get('CHAVIS_REACTION_IDS')
    if requested is None:
        return enabled
    try:
        selected = json.loads(requested)
    except (TypeError, ValueError):
        raise ValueError('CHAVIS_REACTION_IDS must be a JSON list of enabled reaction IDs') from None
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(rid, str) for rid in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('CHAVIS_REACTION_IDS must be a nonempty list of unique reaction IDs')
    unavailable = [rid for rid in selected if rid not in enabled]
    if unavailable:
        raise ValueError(f'CHAVIS_REACTION_IDS includes unknown or disabled reactions: {unavailable}')
    return selected


def load_params(path):
    path = Path(path)
    raw = json.loads(path.read_text()) if path.is_file() else {}
    from chavis.inputs.fs_build import FS_DEFAULTS
    from chavis.search.route_v2 import TS_DEFAULTS
    return {
        **DEFAULT_PARAMS,
        **FS_DEFAULTS,
        **TS_DEFAULTS,
        **{k: v for k, v in raw.items() if not k.startswith('_')}
    }


FALLBACK_ORDER = ('bound', 'floating')


def select_with_fallback(minima, params=None, outcomes=('bound',)):
    chosen, report = select(minima, params, outcomes)
    report['fallback'] = None
    if chosen:
        return (chosen, report)
    present = {m['outcome'] for m in minima}
    for oc in FALLBACK_ORDER:
        if oc in outcomes or oc not in present:
            continue
        chosen, rep = select(minima, params, (oc,))
        if chosen:
            rep['fallback'] = {
                'requested': list(outcomes),
                'used': oc,
                'reason': 'no minimum of the requested outcome exists'
            }
            return (chosen, rep)
    return (chosen, report)


def state_decision(minima, window_eV):
    eb = min((m['energy_eV'] for m in minima if m['outcome'] == 'bound'), default=None)
    ef = min(
        (m['energy_eV'] for m in minima if m['outcome'] == 'floating'),
        default=None
    )
    if eb is None and ef is None:
        return {'lowest': None, 'delta_bound_minus_floating_eV': None, 'both': False}
    lowest = 'bound' if ef is None or (eb is not None and eb < ef) else 'floating'
    delta = None if eb is None or ef is None else round(eb - ef, 4)
    return {
        'lowest': lowest,
        'delta_bound_minus_floating_eV': delta,
        'both': delta is not None and abs(delta) <= float(window_eV) + 1e-12
    }
