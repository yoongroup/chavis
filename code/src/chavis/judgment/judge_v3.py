from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
import numpy as np
from copy import deepcopy

JUDGE_VERSION = 'chavis_common_judge_v3'
IMAGINARY_THRESHOLD_CM1 = 50.0
CONNECTION_INITIAL_STEPS = 500
CONNECTION_TOTAL_STEPS = 1500
DUPLICATE_ENERGY_EV = 0.001
DUPLICATE_MAX_MIC_A = 0.05
TS_FMAX = 0.005
ENDPOINT_FMAX = 0.01
H_MIN_A = 0.0
SLAB_MAX_A = 0.7
MODE_MIN_ABS = 1e-06
MODE_MIN_SHARE = 0.01
MODE_SHARE_FLOOR = 0.01
ENERGY_TOL_EV = 0.01
TS_BOND_MAX_SCALE = 3.0
METASTABLE_EV = 0.1


def imaginary_summary(freqs, *, signed=False, threshold_cm1=IMAGINARY_THRESHOLD_CM1,
                      eps_cm1=1e-6):
    """Classify ASE complex frequencies or an explicitly signed cached spectrum.

    Hard imaginary modes include the exact 50 cm-1 boundary. Soft values are
    retained as negative wavenumbers; they do not change the one-hard-mode gate.
    """
    if not math.isfinite(threshold_cm1) or threshold_cm1 <= eps_cm1:
        raise ValueError('imaginary threshold must be finite and exceed epsilon')
    values = []
    for value in freqs:
        if signed:
            value = float(value)
        else:
            value = complex(value)
            value = -abs(float(value.imag)) if abs(value.imag) > eps_cm1 else float(value.real)
        if not math.isfinite(value):
            raise ValueError('non-finite frequency')
        values.append(value)
    all_imag = [i for i, value in enumerate(values) if value < -eps_cm1]
    hard = [i for i in all_imag if abs(values[i]) >= threshold_cm1]
    return dict(judge_version=JUDGE_VERSION, imaginary_threshold_cm1=float(threshold_cm1),
                hard_indices=hard, imaginary_count=len(hard),
                imaginary_mode_index=hard[0] if len(hard) == 1 else None,
                all_imaginary_count=len(all_imag), signed_cm1=values,
                soft_imaginary=[values[i] for i in all_imag if i not in hard])


def classify_frequency_record(record, spectrum=None):
    """Normalize a cached record using its complete signed frequency spectrum.

    A legacy ``cm1`` preview cannot establish the mode count if it is shorter
    than 3*n_free. Supply the saved *_frequencies_cm1.json instead in that case.
    """
    out = deepcopy(record or {})
    if spectrum is None:
        for key in ('signed_cm1', 'frequencies_cm1', 'cm1'):
            if key in out:
                spectrum = out[key]
                if key == 'cm1' and out.get('n_free') is not None and len(spectrum) < 3 * int(out['n_free']):
                    raise ValueError('truncated frequency preview; complete spectrum required')
                break
    if spectrum is None:
        raise ValueError('frequency spectrum missing; imaginary_count alone is insufficient')
    out.update(imaginary_summary(spectrum, signed=True))
    out['cm1'] = out['signed_cm1'][:12]
    out['review_soft_imaginary'] = bool(out['soft_imaginary'])
    return out


def run_connection_optimizer(opt, *, fmax, initial_steps=CONNECTION_INITIAL_STEPS,
                             total_steps=CONNECTION_TOTAL_STEPS):
    """Continue the same live ASE optimizer; run(steps=...) is incremental."""
    if not 0 < initial_steps <= total_steps:
        raise ValueError('connection limits must satisfy 0 < initial <= total')
    if int(opt.nsteps) != 0:
        raise ValueError('connection optimizer must start at zero steps')
    stages = []
    ok = bool(opt.run(fmax=fmax, steps=initial_steps))
    stages.append(dict(limit=initial_steps, steps=int(opt.nsteps), converged=ok))
    if not ok and int(opt.nsteps) < total_steps:
        ok = bool(opt.run(fmax=fmax, steps=total_steps - int(opt.nsteps)))
        stages.append(dict(limit=total_steps, steps=int(opt.nsteps), converged=ok))
    if int(opt.nsteps) > total_steps:
        raise RuntimeError('connection optimizer exceeded cumulative step limit')
    return ok, dict(judge_version=JUDGE_VERSION, initial_steps=initial_steps,
                    total_steps=total_steps, continued=len(stages) == 2, stages=stages)


def should_stop(validated, share):
    return bool(validated and isinstance(share, (int, float, np.number))
                and not isinstance(share, (bool, np.bool_))
                and math.isfinite(float(share)) and float(share) >= MODE_MIN_SHARE)


class PairJudgmentCache:
    """Exact-pair, atom-order-preserving v3 saddle judgment reuse.

    Caller owns a distinct instance for each endpoint pair. Source judgments
    must have been produced by this module after its frequency/connection policy
    was applied. No transitive clustering: each lookup compares an original TS.
    """

    def __init__(self, pair_key):
        self.pair_key = str(pair_key)
        self.records = []

    @staticmethod
    def _eligible(row):
        d = row.get('dimer') or {}
        return bool(d.get('converged') and d.get('fmax') is not None
                    and float(d['fmax']) <= 0.005 and d.get('curvature') is not None
                    and float(d['curvature']) < 0 and d.get('energy_eV') is not None
                    and math.isfinite(float(d['energy_eV'])))

    def remember(self, ts, row, judged, extra, route, out):
        if ts is None or not self._eligible(row) or row.get('duplicate_of'):
            return
        if judged.get('judge_version') != JUDGE_VERSION:
            raise ValueError('only completed common judge v3 judgments may be cached')
        if judged.get('evaluation_complete_for_reuse') is not True:
            return
        atoms = ts.copy()
        atoms.calc = None
        self.records.append(dict(atoms=atoms, row=deepcopy(row), judged=deepcopy(judged),
                                 extra=deepcopy(extra), route=route, out=Path(out)))

    def reuse(self, ts, row, route, out):
        from ase.geometry import find_mic
        import shutil
        if ts is None or not self._eligible(row):
            return None
        energy = float(row['dimer']['energy_eV'])
        for prior in self.records:
            source, atoms = prior['row'], prior['atoms']
            de = energy - float(source['dimer']['energy_eV'])
            if abs(de) >= DUPLICATE_ENERGY_EV:
                continue
            if (not np.array_equal(ts.numbers, atoms.numbers)
                    or not np.array_equal(ts.pbc, atoms.pbc)
                    or not np.allclose(ts.cell.array, atoms.cell.array, rtol=0, atol=1e-12)):
                continue
            distances = find_mic(ts.positions - atoms.positions, ts.cell, ts.pbc)[1]
            max_dist = float(np.max(distances)) if len(distances) else 0.0
            if max_dist >= DUPLICATE_MAX_MIC_A:
                continue
            result = deepcopy(source)
            for key in ('order', 'seed_id', 'checkpoint', 'stage', 'image'):
                result.pop(key, None)
            result.update(deepcopy(row))
            # A caller may initialize empty validation fields before this
            # lookup. They must not overwrite the reused scientific evidence.
            for key in ('frequencies', 'connection', 'reaction_graph_chemistry',
                        'geometry_precheck', 'reaction_mode', 'connects_reaction_graphs', 'verdict'):
                if key in source:
                    result[key] = deepcopy(source[key])
            for key in ('barrier_forward_eV', 'barrier_reverse_eV'):
                if key not in row and source.get(key) is not None:
                    result[key] = float(source[key]) + de
            result['duplicate_of'] = f"{prior['route']}:{source['seed']}"
            result['duplicate_match'] = dict(pair_key=self.pair_key,
                                             energy_difference_eV=de,
                                             max_atom_MIC_displacement_A=max_dist,
                                             energy_tolerance_eV=DUPLICATE_ENERGY_EV,
                                             displacement_tolerance_A=DUPLICATE_MAX_MIC_A)
            result['cost'] = deepcopy(row.get('cost') or {})
            for stage in ('freq', 'conn'):
                result['cost'][stage] = dict(status='skipped', reason='duplicate_v3_judgment',
                                             calls=0, seconds=0.0)
            judged, extra = deepcopy(prior['judged']), deepcopy(prior['extra'])
            judged.update(seed=row['seed'], route=route, duplicate_of=result['duplicate_of'],
                           duplicate_match=deepcopy(result['duplicate_match']))
            for key in ('barrier_from_pair_IS_eV', 'barrier_from_actual_reactant_eV',
                        'barrier_from_lower_reactant_eV'):
                if judged.get(key) is not None:
                    judged[key] = round(float(judged[key]) + de, 4)
            result['_judgment_v3_reuse'] = dict(judged=judged, extra=extra)
            result['judged_full'] = judged
            result['judged'] = dict(validated=extra['validated'], success=bool(judged.get('success')),
                                    share_raw=extra['share_raw'], failed_gate=judged.get('failed_gate'),
                                    barrier_from_lower_reactant_eV=judged.get('barrier_from_lower_reactant_eV'),
                                    fast_stop_candidate=should_stop(extra['validated'], extra['share_raw']))
            destination = Path(out)
            for suffix in ('imaginary_mode.npy', 'frequencies_cm1.json',
                           'minus_endpoint.extxyz', 'plus_endpoint.extxyz'):
                src = prior['out'] / f"{source['seed']}_{suffix}"
                dst = destination / f"{row['seed']}_{suffix}"
                if src.is_file() and src.resolve() != dst.resolve():
                    shutil.copyfile(src, dst)
            return result
        return None


def pair_cache(P):
    if not hasattr(P, '_judge_v3_cache'):
        P._judge_v3_cache = PairJudgmentCache(str(Path(P.out).resolve()))
    return P._judge_v3_cache


def check_mode_min_share(value):
    message = f'mode_min_share must be a finite number between {MODE_SHARE_FLOOR} and 1.0 (mode-share screening policy)'
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if not math.isfinite(value) or not MODE_SHARE_FLOOR <= value <= 1.0:
        raise ValueError(message)
    return value


def _read(path):
    from ase.io import read
    return read(str(path)) if Path(path).is_file() else None


def equivalent_bonds(sym, bond, g_is):
    pair = sorted((sym[bond[0]], sym[bond[1]]))
    out = {tuple(sorted(e)) for e in g_is if sorted((sym[e[0]], sym[e[1]])) == pair}
    return sorted(out | {tuple(sorted(bond))})


def reaction_bond_check(ts, bond, g_is=()):
    from ase.data import covalent_radii, atomic_numbers
    sym = ts.get_chemical_symbols()
    cands = equivalent_bonds(sym, bond, g_is)
    d, (a, b) = max((
        (float(ts.get_distance(x, y, mic=True)), (x, y)) for x,
        y in cands
    ))
    lim = TS_BOND_MAX_SCALE * float(covalent_radii[atomic_numbers[sym[a]]] + covalent_radii[atomic_numbers[sym[b]]])
    return {
        'ok': d <= lim,
        'bond': [int(a), int(b)],
        'distance_A': round(d, 4),
        'limit_A': round(lim, 4),
        'equivalent_bonds': [list(map(int, c)) for c in cands],
        'reason': None if d <= lim else 'reaction_bond_overstretched'
    }


def geometry(sp, atoms, IS, free_metal):
    from ase.geometry import find_mic
    layers = sp.record['structure']['layers']
    z0 = float(np.mean([sp.clean[i, 2] for i, l in enumerate(layers) if l == 1]))
    h = float(np.min(atoms.positions[sp.n:, 2]) - z0)
    d = find_mic(
        atoms.positions[free_metal] - IS.positions[free_metal],
        IS.cell,
        IS.pbc
    )[1]
    slab = float(d.max()) if len(d) else 0.0
    ok = h >= H_MIN_A and slab <= SLAB_MAX_A
    return {
        'ok': ok,
        'min_ads_height_A': round(h, 4),
        'max_free_metal_disp_A': round(slab, 4),
        'reason': None if ok else 'below_surface' if h < H_MIN_A else 'slab_damage'
    }


def graph(sp, atoms):
    from chavis.judgment.structure_labels import adsorbate_graph
    return {tuple(sorted(e)) for e in adsorbate_graph(
        atoms,
        list(range(sp.n, sp.n + len(sp.sym))),
        sp.rl
    )}


def permutations(sp):
    groups = {}
    for i, s in enumerate(sp.sym):
        groups.setdefault(s, []).append(i + sp.n)
    keys = list(groups)
    for combo in itertools.product(*(itertools.permutations(groups[k]) for k in keys)):
        perm = {}
        for k, p in zip(keys, combo):
            perm.update(dict(zip(groups[k], p)))
        yield perm


def chemistry(sp, g_minus, g_plus, g_is, g_fs):
    mapg = lambda g, p: {tuple(sorted((p[a], p[b]))) for a, b in g}
    for perm in permutations(sp):
        for a, b, orient in (
            (g_minus, g_plus, {'minus': 'IS', 'plus': 'FS'}),
            (g_plus, g_minus, {'plus': 'IS', 'minus': 'FS'})
        ):
            if mapg(a, perm) == g_is and mapg(b, perm) == g_fs:
                return {
                    'ok': True,
                    'orientation': orient,
                    'permutation': {int(k): int(v) for k, v in perm.items() if k != v}
                }
    return {'ok': False, 'orientation': None, 'permutation': None}


def mode_metrics(sp, ts, mode, bond, free, min_share=MODE_MIN_SHARE, with_raw=False):
    min_share = check_mode_min_share(min_share)
    m = np.array(mode, float)
    m[[i for i in range(len(m)) if i not in set(free)]] = 0
    m /= np.linalg.norm(m)
    a, b = bond
    u = ts.get_distance(a, b, mic=True, vector=True)
    u /= np.linalg.norm(u)
    proj = float(np.dot(m[b] - m[a], u))
    share = proj ** 2 / 2.0
    rel = m[b] - m[a]
    cos = float(abs(proj) / np.linalg.norm(rel)) if np.linalg.norm(rel) > 1e-12 else 0.0
    ads = list(range(sp.n, sp.n + len(sp.sym)))
    stretches = {}
    for e in graph(sp, ts) | {tuple(sorted(bond))}:
        v = ts.get_distance(*e, mic=True, vector=True)
        v /= np.linalg.norm(v)
        stretches[e] = abs(float(np.dot(m[e[1]] - m[e[0]], v)))
    rank = 1 + sum((
        1 for e,
        s in stretches.items() if s > stretches[tuple(sorted(bond))] + 1e-12
    ))
    out = {
        'ok': bool(share >= min_share),
        'min_share': min_share,
        'bond': [int(a), int(b)],
        'projection': round(proj, 5),
        'bond_stretch_share': round(share, 4),
        'cos_relative_motion_to_bond': round(cos, 4),
        'rank_among_adsorbate_bonds': rank,
        'n_adsorbate_bonds': len(stretches)
    }
    return (out, float(share)) if with_raw else out


def pair_context(sp, pair_dir, res=None):
    pair_dir = Path(pair_dir)
    res = res if res is not None else json.loads((pair_dir / 'result.json').read_text())
    IS, FS = (_read(pair_dir / 'IS.extxyz'), _read(pair_dir / 'FS.extxyz'))
    if IS is None or FS is None:
        return None
    from chavis.surface.metal_mode import fixed_indices
    from chavis.inputs.minima import reactions_db
    fixed = set(fixed_indices(sp.record, res['fixed_layers']))
    free = [i for i in range(len(IS)) if i not in fixed]
    rx = reactions_db().get(res['reaction'], {})
    state_class = str(rx.get('state_class') or 'adsorbed_to_adsorbed')
    g_is, g_fs = (graph(sp, IS), graph(sp, FS))
    lost = sorted(g_is - g_fs)
    return {
        'res': res,
        'IS': IS,
        'FS': FS,
        'free': free,
        'free_metal': [i for i in free if i < sp.n],
        'need_adsorbed_products': state_class.endswith('_to_adsorbed'),
        'need_adsorbed_reactant': state_class.startswith('adsorbed_to_'),
        'g_is': g_is,
        'g_fs': g_fs,
        'bond': lost[0] if len(lost) == 1 else None
    }


def judge_seed(sp, pair_dir, route_name, s, ctx, mode_min_share=MODE_MIN_SHARE):
    from chavis.judgment.structure_labels import label as struct_label
    pair_dir = Path(pair_dir)
    mode_min_share = check_mode_min_share(mode_min_share)
    reused = s.get('_judgment_v3_reuse')
    if reused is not None:
        judged, extra = deepcopy(reused['judged']), deepcopy(reused['extra'])
        if judged.get('judge_version') != JUDGE_VERSION or not s.get('duplicate_of'):
            raise ValueError('invalid duplicate judgment provenance')
        if TS_FMAX != 0.005 or (judged.get('reaction_mode') or {}).get('min_share', mode_min_share) != mode_min_share:
            raise ValueError('cannot reuse duplicate judgment under changed gates')
        return judged, extra
    s = deepcopy(s)
    if s.get('frequencies'):
        full_path = pair_dir / f"{s['seed']}_frequencies_cm1.json"
        spectrum = json.loads(full_path.read_text()) if full_path.is_file() else None
        s['frequencies'] = classify_frequency_record(s['frequencies'], spectrum=spectrum)
    res, IS, free, free_metal = (ctx['res'], ctx['IS'], ctx['free'], ctx['free_metal'])
    g_is, g_fs, bond = (ctx['g_is'], ctx['g_fs'], ctx['bond'])
    need_adsorbed_products, need_adsorbed_reactant = (
        ctx['need_adsorbed_products'],
        ctx['need_adsorbed_reactant']
    )
    share_raw = None
    label = s['seed']
    d = s['dimer']
    row = {'route': route_name, 'seed': label, 'judge_version': JUDGE_VERSION}
    phys = {
        'converged': bool(d.get('converged') and d.get('fmax') is not None and (d['fmax'] <= TS_FMAX) and (d.get('curvature') is not None) and (d['curvature'] < 0)),
        'imaginary_count': (s.get('frequencies') or {}).get('imaginary_count'),
        'imaginary_threshold_cm1': IMAGINARY_THRESHOLD_CM1,
        'soft_imaginary': (s.get('frequencies') or {}).get('soft_imaginary', [])
    }
    ts = _read(pair_dir / f'{label}_ts.extxyz')
    arrivals = {k: _read(pair_dir / f'{label}_{k}_endpoint.extxyz') for k in ('minus', 'plus')}
    chem = chemistry(
        sp,
        graph(sp, arrivals['minus']),
        graph(sp, arrivals['plus']),
        g_is,
        g_fs
    ) if ts is not None and all(arrivals.values()) else None
    mapped = bool(chem and chem['ok'])
    inv = {int(v): int(k) for k, v in ((chem or {}).get('permutation') or {}).items()}
    ts_bond = (inv.get(bond[0], bond[0]), inv.get(bond[1], bond[1])) if bond else None
    phys['geometry'] = geometry(sp, ts, IS, free_metal) if ts is not None else None
    phys['reaction_bond'] = reaction_bond_check(
        ts,
        ts_bond if mapped else bond,
        () if mapped else g_is
    ) if ts is not None and bond else None
    if phys['reaction_bond'] is not None:
        phys['reaction_bond'].update(
            bond_in_pair_IS_numbering=[int(bond[0]), int(bond[1])],
            permutation_applied=bool(inv),
            bond_reference='chemistry_permutation' if mapped else 'pair_IS_equivalents_fallback',
            fallback_reason=None if mapped else 'endpoints_missing' if chem is None else 'chemistry_unmatched'
        )
    phys['ok'] = bool(phys['converged'] and phys['imaginary_count'] == 1 and phys['geometry'] and phys['geometry']['ok'] and (phys['reaction_bond'] is None or phys['reaction_bond']['ok']))
    row['physical'] = phys
    conn = {c['side']: c for c in s.get('connection', [])}
    if phys['ok'] and all(arrivals.values()) and (len(conn) == 2):
        down = {k: {
            'converged': bool(conn[k]['converged'] and conn[k]['fmax'] <= ENDPOINT_FMAX),
            'fmax': conn[k]['fmax'],
            'energy_eV': conn[k]['energy_eV'],
            'geometry': geometry(sp, arrivals[k], IS, free_metal)
        } for k in ('minus', 'plus')}
        down_ok = all((v['converged'] and v['geometry']['ok'] for v in down.values()))
        row['downhill'] = {'ok': down_ok, **down}
        row['chemistry'] = chem
        mode_path = pair_dir / f'{label}_imaginary_mode.npy'
        if mode_path.is_file() and bond:
            row['reaction_mode'], share_raw = mode_metrics(
                sp,
                ts,
                np.load(mode_path),
                ts_bond,
                free,
                mode_min_share,
                with_raw=True
            )
        else:
            row['reaction_mode'] = {
                'ok': False,
                'reason': 'mode_or_bond_missing',
                'min_share': mode_min_share
            }
        if bond:
            row['reaction_mode']['bond_in_pair_IS_numbering'] = [int(bond[0]), int(bond[1])]
            row['reaction_mode']['permutation_applied'] = bool(inv)
        same_site = any((conn[k]['IS']['match'] for k in conn)) and any((conn[k]['FS']['match'] for k in conn))
        row['same_site_channel_as_pair'] = bool(same_site)
        e_ts = d['energy_eV']
        if chem['ok']:
            r_side = [k for k, v in chem['orientation'].items() if v == 'IS'][0]
            p_side = [k for k, v in chem['orientation'].items() if v == 'FS'][0]
            row['barrier_from_actual_reactant_eV'] = round(e_ts - down[r_side]['energy_eV'], 4)
            row['actual_reactant_above_pair_IS_eV'] = round(
                down[r_side]['energy_eV'] - res['E_IS_eV'],
                4
            )
            row['actual_product_above_pair_FS_eV'] = round(
                down[p_side]['energy_eV'] - res['E_FS_eV'],
                4
            )
            lower_is_pair = res['E_IS_eV'] <= down[r_side]['energy_eV']
            row['barrier_from_lower_reactant_eV'] = round(
                e_ts - min(res['E_IS_eV'], down[r_side]['energy_eV']),
                4
            )
            row['barrier_reference'] = 'pair_IS' if lower_is_pair else 'arrived_reactant'
            row['metastable_reactant'] = bool(down[r_side]['energy_eV'] - res['E_IS_eV'] > METASTABLE_EV)

            def detached(k):
                lab = struct_label(
                    arrivals[k],
                    list(range(sp.n)),
                    list(range(sp.n, len(arrivals[k]))),
                    reference_edges=sp.ref,
                    reference_bond_lengths=sp.rl,
                    fixed_slab_indices=sp.normal_fixed
                )
                return lab['desorption']['detached_fragments']
            row['product_side_detached_fragments'] = detached(p_side)
            row['reactant_side_detached_fragments'] = detached(r_side)
            row['required_surface_state_ok'] = bool(not (need_adsorbed_products and row['product_side_detached_fragments']) and (not (need_adsorbed_reactant and row['reactant_side_detached_fragments'])))
        row['barrier_from_pair_IS_eV'] = round(e_ts - res['E_IS_eV'], 4)
        row['energy_review'] = [k for k in ('minus', 'plus') if down[k]['energy_eV'] > e_ts]
        row['energy_consistent'] = all((down[k]['energy_eV'] <= e_ts + ENERGY_TOL_EV for k in ('minus', 'plus')))
        row['warnings'] = [] if row['energy_consistent'] else [f"arrival above TS by more than {ENERGY_TOL_EV} eV: {row['energy_review']}"]
        row['success'] = bool(down_ok and row['reaction_mode']['ok'] and chem['ok'])
        row['validated_success'] = bool(row['success'] and row.get('required_surface_state_ok', False))
    else:
        row['success'] = row['validated_success'] = False
    row['failed_gate'] = None if row['success'] else ('physical:reaction_bond_overstretched' if (phys.get('reaction_bond') or {}).get('ok') is False else 'physical') if not phys['ok'] else 'downhill' if not row.get('downhill', {}).get('ok') else 'reaction_mode' if not row['reaction_mode']['ok'] else 'chemistry'
    # A legacy 500-step failed connection is not a v3-complete judgment. It must
    # receive the requested continuation before it can seed the duplicate cache.
    connection_complete = len(conn) == 2 and all(
        (c.get('converged') and c.get('fmax') is not None and c['fmax'] <= ENDPOINT_FMAX)
        or int(c.get('steps') or 0) >= CONNECTION_TOTAL_STEPS for c in conn.values())
    precheck_failed = ((phys.get('geometry') or {}).get('ok') is False
                       or (phys.get('reaction_bond') or {}).get('ok') is False)
    physical_evaluated = bool(phys['converged'] and phys.get('geometry') is not None
                              and (precheck_failed or phys['imaginary_count'] is not None))
    row['evaluation_complete_for_reuse'] = bool(physical_evaluated
                                               and (not phys['ok'] or connection_complete))
    return (
        row,
        {'share_raw': share_raw, 'validated': bool(row.get('validated_success'))}
    )


def judge_pair(sp, pair_dir, mode_min_share=MODE_MIN_SHARE):
    mode_min_share = check_mode_min_share(mode_min_share)
    pair_dir = Path(pair_dir)
    res = json.loads((pair_dir / 'result.json').read_text())
    ctx = pair_context(sp, pair_dir, res)
    if ctx is None:
        return {'status': 'structures_not_saved', 'pair_dir': str(pair_dir)}
    bond = ctx['bond']
    seeds = [judge_seed(
        sp,
        pair_dir,
        route['route'],
        s,
        ctx,
        mode_min_share
    )[0] for route in res['routes'] for s in route['seeds']]
    ok = [r for r in seeds if r['success']]
    val = [r for r in seeds if r.get('validated_success')]
    out = {
        'judge_version': JUDGE_VERSION,
        'reaction': res['reaction'],
        'equation': res['equation'],
        'pair_group': res['pair_group'],
        'validated_success': bool(val),
        'n_validated_seeds': len(val),
        'lowest_barrier_from_lower_reactant_eV': min(
            (r['barrier_from_lower_reactant_eV'] for r in val),
            default=None
        ),
        'lowest_barrier_seed': min(
            ((r['barrier_from_lower_reactant_eV'], r['seed']) for r in val),
            default=(None, None)
        )[1],
        'warnings': sorted({w for r in seeds for w in r.get('warnings', [])}),
        'paired_is': res['paired_is'],
        'fs_source': res['fs_source'],
        'reaction_bond': list(bond) if bond else None,
        'success': bool(ok),
        'n_seeds': len(seeds),
        'n_success_seeds': len(ok),
        'same_site_success': any((r.get('same_site_channel_as_pair') for r in ok)),
        'lowest_barrier_from_actual_reactant_eV': min(
            (r['barrier_from_actual_reactant_eV'] for r in ok),
            default=None
        ),
        'lowest_barrier_from_pair_IS_eV': min(
            (r['barrier_from_pair_IS_eV'] for r in ok),
            default=None
        ),
        'criteria': {
            'ts_fmax': TS_FMAX,
            'ts_bond_max_scale': TS_BOND_MAX_SCALE,
            'endpoint_fmax': ENDPOINT_FMAX,
            'h_min_A': H_MIN_A,
            'slab_max_A': SLAB_MAX_A,
            'mode_min_abs': MODE_MIN_ABS,
            'mode_min_share': mode_min_share,
            'mode_share_floor': MODE_SHARE_FLOOR,
            'mode_rule': 'bond_stretch_share >= mode_min_share',
            'imaginary_threshold_cm1': IMAGINARY_THRESHOLD_CM1,
            'imaginary_rule': 'exactly one imaginary mode with absolute frequency >= 50 cm-1',
            'connection_steps': [CONNECTION_INITIAL_STEPS, CONNECTION_TOTAL_STEPS],
            'source': JUDGE_VERSION
        },
        'seeds': seeds
    }
    (pair_dir / 'judgment.json').write_text(json.dumps(out, indent=1))
    return out
