from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
TS_DEFAULTS = {
    'ts_fix_layers': '3,4',
    'dimer_max_steps': 1500,
    'ts_route_override': None
}
TS_MODES = ('fast', 'slow')
FAST_STOP_SHARE = 0.01
CANDIDATE_ORDER = 'direct>scan_stage_a_peak_outward>scan_stage_b v1'
TS_SETTINGS = {
    'images': 9,
    'idpp_policy': 'ni_drm_mic_idpp_v1 (FIRE fmax 0.05, 150 steps)',
    'neb': {
        'method': 'improvedtangent',
        'optimizer': 'FIRE',
        'dt': 0.05,
        'maxstep_A': 0.1,
        'warm_fmax': 0.1,
        'ci_fmax': 0.03,
        'steps': 500
    },
    'scan': {
        'step_A': 0.1,
        'optimizer': 'BFGS',
        'maxstep_A': 0.1,
        'fmax': 0.05,
        'steps': 300
    },
    'dimer_fmax': 0.005,
    'freq_delta_A': 0.01,
    'freq_nfree': 2,
    'imag_eps_cm1': 1e-06,
    'imag_soft_cm1': 50.0,
    'connection_displacement_A': 0.2,
    'endpoint': {'maxstep_A': 0.15, 'fmax': 0.01, 'steps': 500},
    'connection_max_steps': 1500,
    'judge_version': 'chavis_common_judge_v3',
    'match_rms_A': 0.25,
    'collision_scale': 0.7,
    'mode_projection_min_abs': 1e-06
}


def clean_ts_params(raw):
    from chavis.surface.metal_mode import parse
    p = {**TS_DEFAULTS, **{k: v for k, v in (raw or {}).items() if k in TS_DEFAULTS}}
    parse(p['ts_fix_layers'])
    p['dimer_max_steps'] = int(p['dimer_max_steps'])
    if p['dimer_max_steps'] < 1:
        raise ValueError('dimer_max_steps must be >= 1')
    if p['ts_route_override'] not in (None, 'direct_only', 'scan_only', 'scan_then_direct'):
        raise ValueError('ts_route_override must be null, direct_only, scan_only or scan_then_direct')
    return p


def check_mode(mode, light=False):
    if mode is None:
        return None
    if mode not in TS_MODES:
        raise ValueError(f'mode must be one of {TS_MODES} or None, got {mode!r}')
    if light:
        raise ValueError('mode and light cannot be combined (light keeps its own stop rule)')
    return mode


class Pair:

    def __init__(self, sp, pair, reaction, fix_layers, out):
        from ase.constraints import FixAtoms
        from chavis.surface.metal_mode import fixed_indices
        self.sp, self.pair, self.reaction = (sp, pair, reaction)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.fixed = fixed_indices(sp.record, fix_layers)
        self.fix_layers = fix_layers
        n = sp.n
        self.ads = list(range(n, n + len(sp.sym)))
        self.IS = sp.atoms(pair['is_slab_positions'], pair['is_adsorbate_positions'])
        self.FS = sp.atoms(pair['fs_slab_positions'], pair['fs_adsorbate_positions'])
        from ase.geometry import find_mic
        diff = find_mic(
            self.FS.positions[self.fixed] - self.IS.positions[self.fixed],
            self.IS.cell,
            self.IS.pbc
        )[1]
        self.fixed_snap_A = float(diff.max()) if len(diff) else 0.0
        if self.fixed_snap_A > 0.05:
            raise ValueError(f'fixed atoms differ between IS and FS by {self.fixed_snap_A:.3f} A')
        self.FS.positions[self.fixed] = self.IS.positions[self.fixed]
        for a in (self.IS, self.FS):
            a.set_constraint(FixAtoms(indices=self.fixed))
            a.info['adsorbate_indices'] = self.ads
        self.free = [i for i in range(len(self.IS)) if i not in set(self.fixed)]
        self.log = []

    def calc(self):
        from chavis.inputs.is_from_fs import _float64
        return _float64(self.sp.ensure_calc())

    def energy(self, atoms):
        a = atoms.copy()
        a.calc = self.calc()
        return float(a.get_potential_energy())

    def relax(self, atoms, label, *, connection=False):
        from ase.optimize import BFGS
        a = atoms.copy()
        a.info = dict(atoms.info)
        a.calc = self.calc()
        s = TS_SETTINGS['endpoint']
        opt = BFGS(a, maxstep=s['maxstep_A'], logfile=str(self.out / f'{label}.log'))
        continuation = None
        if connection:
            from chavis.judgment.judge_v3 import run_connection_optimizer
            ok, continuation = run_connection_optimizer(
                opt, fmax=s['fmax'], initial_steps=s['steps'],
                total_steps=TS_SETTINGS['connection_max_steps'])
        else:
            ok = bool(opt.run(fmax=s['fmax'], steps=s['steps']))
        return (
            a,
            {
                'converged': ok,
                'continuation': continuation,
                'steps': int(opt.nsteps),
                'energy_eV': float(a.get_potential_energy()),
                'fmax': float(np.linalg.norm(a.get_forces(), axis=1).max())
            }
        )


def save(P, name, atoms):
    from ase.io import write
    items = atoms if isinstance(atoms, list) else [atoms]
    out = []
    for a in items:
        b = a.copy()
        b.info = {k: v for k, v in a.info.items() if isinstance(v, (int, float, str))}
        try:
            b.info['energy_eV'] = float(a.get_potential_energy()) if a.calc is not None else None
        except Exception:
            b.info['energy_eV'] = None
        if b.info['energy_eV'] is None:
            b.info.pop('energy_eV')
        out.append(b)
    write(str(P.out / name), out, format='extxyz')


def _structure(atoms, fixed, ads):
    from chavis.search.geometry import from_ase
    s = from_ase(atoms)
    s['fixed_indices'] = list(map(int, fixed))
    s['adsorbate_indices'] = list(ads)
    s['meta'] = {**s.get('meta', {}), 'adsorbate_indices': list(ads)}
    return s


def match(P, endpoint, ref, ops, sperm):
    from scipy.optimize import linear_sum_assignment
    from ase.geometry import find_mic
    from chavis.judgment.structure_labels import adsorbate_graph
    from chavis.inputs.pairs import _groups, _mic
    sp = P.sp
    n = sp.n
    cell = sp.slab.cell.array
    ge = {tuple(sorted(e)) for e in adsorbate_graph(endpoint, P.ads, sp.rl)}
    gr = {tuple(sorted(e)) for e in adsorbate_graph(ref, P.ads, sp.rl)}
    ea, ra = (endpoint.positions[n:], ref.positions[n:])
    es, rs = (endpoint.positions[:n], ref.positions[:n])
    free_metal = [i for i in P.free if i < n]
    best = None
    for oi, (r, t) in enumerate(ops):
        m = ea @ r.T + t
        perm = np.empty(len(sp.sym), int)
        tot = 0.0
        for idx in _groups(sp.sym):
            c = _mic(m[idx], ra[idx], cell)
            ri, ci = linear_sum_assignment(c)
            tot += float(np.sum(c[ri, ci] ** 2))
            perm[np.asarray(idx)[ri]] = np.asarray(idx)[ci]
        ads_rms = (tot / len(sp.sym)) ** 0.5
        mapped = {
            tuple(sorted((int(perm[a - n]) + n, int(perm[b - n]) + n))) for a,
            b in ge
        }
        ms = es @ r.T + t
        d_metal = find_mic(
            ms[free_metal] - rs[sperm[oi][free_metal]],
            cell,
            [True, True, False]
        )[1]
        free_rms = float(np.sqrt((np.sum(d_metal ** 2) + tot) / (len(free_metal) + len(sp.sym))))
        row = {
            'op': oi,
            'ads_rms_A': round(ads_rms, 4),
            'free_rms_A': round(free_rms, 4),
            'edges_match': mapped == gr,
            'permutation': perm.tolist()
        }
        row['match'] = bool(row['edges_match'] and ads_rms <= TS_SETTINGS['match_rms_A'] and (free_rms <= TS_SETTINGS['match_rms_A']))
        key = (not row['match'], not row['edges_match'], ads_rms, free_rms)
        if best is None or key < best[0]:
            best = (key, row)
    row = best[1]
    return row


def dimer(P, seed, direction, label, max_steps):
    from ase.constraints import FixAtoms
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
    ts = seed.copy()
    ts.info = dict(seed.info)
    ts.set_constraint(FixAtoms(indices=P.fixed))
    ts.calc = P.calc()
    v = np.array(direction, float)
    v[P.fixed] = 0
    if np.linalg.norm(v) < 1e-12:
        return (None, {'converged': False, 'reason': 'zero_direction'})
    v /= np.linalg.norm(v)
    with DimerControl(
        logfile=str(P.out / f'{label}_control.log'),
        eigenmode_logfile=None,
        initial_eigenmode_method='displacement',
        displacement_method='vector',
        mask=[i in set(P.free) for i in range(len(ts))]
    ) as control:
        mma = MinModeAtoms(ts, control=control, eigenmodes=[v])
        with MinModeTranslate(mma, logfile=str(P.out / f'{label}.log')) as opt:
            try:
                ok = bool(opt.run(fmax=TS_SETTINGS['dimer_fmax'], steps=max_steps))
                err = None
            except Exception as exc:
                ok, err = (False, f'{type(exc).__name__}: {exc}')
            steps = int(opt.nsteps)
        curv = float(mma.get_curvature()) if err is None else None
    fmax = float(np.linalg.norm(ts.get_forces(), axis=1).max()) if err is None else None
    conv = bool(ok and fmax is not None and (fmax <= TS_SETTINGS['dimer_fmax']) and (curv is not None) and (curv < 0))
    save(P, f'{label}_ts.extxyz', ts)
    return (
        ts,
        {
            'converged': conv,
            'optimizer_converged': ok,
            'error': err,
            'steps': steps,
            'fmax': fmax,
            'curvature': curv,
            'energy_eV': float(ts.get_potential_energy()) if err is None else None
        }
    )


def frequencies(P, ts, label):
    from ase.vibrations import Vibrations
    a = ts.copy()
    a.info = dict(ts.info)
    a.calc = P.calc()
    vib = Vibrations(
        a,
        indices=P.free,
        name=str(P.out / f'vib_{label}'),
        delta=TS_SETTINGS['freq_delta_A'],
        nfree=TS_SETTINGS['freq_nfree']
    )
    vib.run()
    f = vib.get_frequencies()
    from chavis.judgment.judge_v3 import imaginary_summary
    summary = imaginary_summary(f, threshold_cm1=TS_SETTINGS['imag_soft_cm1'],
                                eps_cm1=TS_SETTINGS['imag_eps_cm1'])
    imag = summary['hard_indices']
    signed = summary['signed_cm1']
    mode = vib.get_mode(imag[0]) if len(imag) == 1 else None
    if mode is not None:
        np.save(P.out / f'{label}_imaginary_mode.npy', mode)
    (P.out / f'{label}_frequencies_cm1.json').write_text(json.dumps(signed))
    vib.clean()
    return (
        mode,
        {
            **summary,
            'review_soft_imaginary': bool(summary['soft_imaginary']),
            'cm1': signed[:12],
            'n_free': len(P.free)
        }
    )


def soft_imaginary_review(freqs):
    im = sorted(
        (float(x.imag) for x in freqs if x.imag > TS_SETTINGS['imag_eps_cm1']),
        reverse=True
    )
    return bool(len(im) >= 2 and im[0] >= TS_SETTINGS['imag_soft_cm1'] and (im[1] < TS_SETTINGS['imag_soft_cm1']))


def connect(P, ts, mode, label, ops, sperm, chemistry_out=None):
    mode = np.array(mode, float)
    mode[P.fixed] = 0
    mode /= np.linalg.norm(mode, axis=1).max()
    out, arrivals = ([], {})
    for sign, side in ((-1, 'minus'), (1, 'plus')):
        a = ts.copy()
        a.info = dict(ts.info)
        a.positions += sign * TS_SETTINGS['connection_displacement_A'] * mode
        end, diag = P.relax(a, f'{label}_{side}', connection=True)
        arrivals[side] = end
        save(P, f'{label}_{side}_endpoint.extxyz', end)
        out.append({
            'side': side,
            **diag,
            'IS': match(P, end, P.IS, ops, sperm),
            'FS': match(P, end, P.FS, ops, sperm)
        })
    ok = lambda r, k: r['converged'] and r[k]['match']
    connected = ok(out[0], 'IS') and ok(out[1], 'FS') or (ok(out[1], 'IS') and ok(out[0], 'FS'))
    g = lambda r, k: r['converged'] and r[k]['edges_match']
    graphs = g(out[0], 'IS') and g(out[1], 'FS') or (g(out[1], 'IS') and g(out[0], 'FS'))
    if chemistry_out is not None:
        from chavis.judgment.judge_v3 import chemistry, graph
        chemistry_out.update(chemistry(
            P.sp,
            graph(P.sp, arrivals['minus']),
            graph(P.sp, arrivals['plus']),
            graph(P.sp, P.IS),
            graph(P.sp, P.FS)
        ))
    return (connected, graphs, out)


class _Step:

    def __init__(self, P, cost, name):
        self.P, self.cost, self.name = (P, cost, name)

    def __enter__(self):
        self.c0, self.t0 = (self.P.sp.factory.calls, time.monotonic())
        return self

    def __exit__(self, *exc):
        dc = self.P.sp.factory.calls - self.c0
        if dc < 0:
            raise RuntimeError(f'model call counter decreased during {self.name}')
        self.cost[self.name] = {
            'status': 'ran' if exc[0] is None else 'error',
            'calls': dc,
            'seconds': round(time.monotonic() - self.t0, 3)
        }
        return False


def _skipped(reason):
    return {'status': 'skipped', 'reason': reason, 'calls': 0, 'seconds': 0.0}


def validate(P, seed, direction, label, p, ops, sperm, e_is, e_fs):
    modal = p.get('mode') is not None
    route = 'direct' if label.startswith('direct') else 'scan'

    def finish(row, ts):
        from chavis.judgment.judge_v3 import judge_seed, pair_context, pair_cache
        ctx = (p.get('_rt') or {}).get('ctx')
        if ctx is None:
            ctx = pair_context(P.sp, P.out, res=dict(
                reaction=P.reaction['id'], fixed_layers=P.fix_layers,
                E_IS_eV=e_is, E_FS_eV=e_fs))
        if ctx is None:
            raise RuntimeError('IS/FS not saved before common v3 judgment')
        judged, extra = judge_seed(P.sp, P.out, route, row, ctx)
        row['judged_full'] = judged
        pair_cache(P).remember(ts, row, judged, extra, route, P.out)
        return row, ts

    if not modal:
        return finish(*_validate_steps(
            P,
            seed,
            direction,
            label,
            p,
            ops,
            sperm,
            e_is,
            e_fs,
            False,
            {}
        ))
    cost = {}
    try:
        return finish(*_validate_steps(
            P,
            seed,
            direction,
            label,
            p,
            ops,
            sperm,
            e_is,
            e_fs,
            True,
            cost
        ))
    except BaseException as exc:
        rec = {'seed': label, 'cost': cost, 'error': f'{type(exc).__name__}: {exc}'}
        try:
            (P.out / f'{label}_candidate_error.json').write_bytes(json.dumps(rec, indent=1, default=str).encode('utf-8'))
        except OSError:
            pass
        try:
            exc.chavis_candidate = rec
        except AttributeError:
            pass
        raise


def _validate_steps(P, seed, direction, label, p, ops, sperm, e_is, e_fs, modal, cost):
    if modal:
        with _Step(P, cost, 'dimer'):
            ts, d = dimer(P, seed, direction, label, p['dimer_max_steps'])
    else:
        ts, d = dimer(P, seed, direction, label, p['dimer_max_steps'])
    row = {'seed': label, 'dimer': d}
    if modal:
        row['cost'] = cost
    if not d['converged']:
        if modal:
            cost.update(
                freq=_skipped('dimer_not_converged'),
                conn=_skipped('dimer_not_converged')
            )
        row['verdict'] = 'dimer_not_converged'
        return (row, None)
    row['barrier_forward_eV'] = d['energy_eV'] - e_is
    row['barrier_reverse_eV'] = d['energy_eV'] - e_fs
    from chavis.judgment.judge_v3 import pair_cache
    route = 'direct' if label.startswith('direct') else 'scan'
    duplicate = pair_cache(P).reuse(ts, row, route, P.out)
    if duplicate is not None:
        return duplicate, ts
    if modal:
        from chavis.judgment.judge_v3 import geometry
        free_metal = [i for i in P.free if i < P.sp.n]
        geo = geometry(P.sp, ts, P.IS, free_metal)
        row['geometry_precheck'] = geo
        if not geo['ok']:
            cost.update(
                freq=_skipped(f"geometry_{geo['reason']}"),
                conn=_skipped(f"geometry_{geo['reason']}")
            )
            row['verdict'] = 'geometry_rejected'
            return (row, ts)
        with _Step(P, cost, 'freq'):
            mode, fr = frequencies(P, ts, label)
    else:
        mode, fr = frequencies(P, ts, label)
    row['frequencies'] = fr
    if mode is None:
        if modal:
            cost['conn'] = _skipped(f"imaginary_{fr['imaginary_count']}")
        row['verdict'] = f"imaginary_{fr['imaginary_count']}"
        return (row, ts)
    chem = {}
    if modal:
        with _Step(P, cost, 'conn'):
            connected, graphs, conn = connect(
                P,
                ts,
                mode,
                label,
                ops,
                sperm,
                chemistry_out=chem
            )
    else:
        connected, graphs, conn = connect(
            P,
            ts,
            mode,
            label,
            ops,
            sperm,
            chemistry_out=chem
        )
    row['connection'] = conn
    row['reaction_graph_chemistry'] = chem or None
    row['reaction_mode'] = reaction_mode(
        P,
        ts,
        mode,
        permutation=chem.get('permutation') if chem.get('ok') else None
    )
    row['connects_reaction_graphs'] = bool(graphs)
    along = row['reaction_mode']['along_bond']
    row['verdict'] = 'mode_not_along_reaction_bond' if (connected or graphs) and (not along) else 'target_ts' if connected else 'reaction_ts_other_endpoints' if graphs else 'connection_unresolved'
    return (row, ts)


def scan_candidate_order(sel):
    peak = sel.get('peak')
    a = list(sel.get('stage_a') or [])
    if peak is not None:
        a.sort(key=lambda i: (abs(i - peak), i))
    return [('stage_a', a), ('stage_b', list(sel.get('stage_b') or []))]


def _after_candidate(P, p, route, row):
    rt = p['_rt']
    from chavis.judgment.judge_v3 import judge_seed, MODE_MIN_SHARE, should_stop
    judged, extra = judge_seed(P.sp, P.out, route, row, rt['ctx'], MODE_MIN_SHARE)
    share = extra['share_raw']
    confident = should_stop(extra['validated'], share)
    rt['order'] += 1
    row['seed_id'] = f"{route}:{row['seed']}"
    row['order'] = rt['order']
    row['judged'] = {
        'validated': extra['validated'],
        'success': bool(judged.get('success')),
        'share_raw': share,
        'failed_gate': judged.get('failed_gate'),
        'barrier_from_lower_reactant_eV': judged.get('barrier_from_lower_reactant_eV'),
        'fast_stop_candidate': confident
    }
    row['checkpoint'] = {
        'calls_through': P.sp.factory.calls - rt['c0'],
        'seconds_through': round(time.monotonic() - rt['t0'], 3)
    }
    if confident and rt['fast_stop'] is None:
        rt['fast_stop'] = {
            'seed_id': row['seed_id'],
            'order': row['order'],
            **row['checkpoint']
        }
    with open(P.out / 'candidates.jsonl', 'ab') as fh:
        fh.write((json.dumps({'route': route, **row}, default=float) + '\n').encode('utf-8'))
    return confident and p['mode'] == 'fast'


def reaction_mode(P, ts, mode, permutation=None):
    a, b = P.pair.get('bond_cut') or _cut_bond(P)
    pair_bond = [int(a), int(b)]
    inv = {int(v): int(k) for k, v in (permutation or {}).items()}
    a, b = (inv.get(a, a), inv.get(b, b))
    m = np.array(mode, float)
    m[P.fixed] = 0
    m /= np.linalg.norm(m, axis=1).max()
    unit = ts.get_distance(a, b, mic=True, vector=True)
    unit /= np.linalg.norm(unit)
    rel = m[b] - m[a]
    proj = float(np.dot(rel, unit))
    cos = float(abs(proj) / np.linalg.norm(rel)) if np.linalg.norm(rel) > 1e-12 else 0.0
    return {
        'bond': [int(a), int(b)],
        'bond_projection_per_unit_amplitude': round(proj, 4),
        'bond_in_pair_IS_numbering': pair_bond,
        'permutation_applied': bool(inv),
        'cos_relative_motion_to_bond': round(cos, 4),
        'threshold': TS_SETTINGS['mode_projection_min_abs'],
        'along_bond': abs(proj) > TS_SETTINGS['mode_projection_min_abs']
    }


def direct_route(P, p, ops, sperm, e_is, e_fs):
    from ase.mep import NEB
    from ase.optimize import FIRE
    from ase.geometry import find_mic
    from chavis.search.mic_idpp import prepare_idpp, mic_collision_pairs, PathPreparationError
    from chavis.search.geometry import to_ase, from_ase
    s = TS_SETTINGS['neb']
    res = {'route': 'direct'}
    try:
        frames, diag = prepare_idpp(
            _structure(P.IS, P.fixed, P.ads),
            _structure(P.FS, P.fixed, P.ads),
            TS_SETTINGS['images']
        )
    except PathPreparationError as exc:
        res.update(status='idpp_failure', reason=str(exc))
        return (res, [])
    res['idpp'] = diag.get('idpp')
    images = [to_ase(f) for f in frames]
    for a in images:
        a.calc = P.calc()
    neb = NEB(images, climb=False, method=s['method'])
    stages = {}
    for label, fmax, climb in (
        ('warm', s['warm_fmax'], False),
        ('ci', s['ci_fmax'], True)
    ):
        neb.climb = climb
        opt = FIRE(
            neb,
            logfile=str(P.out / f'neb_{label}.log'),
            dt=s['dt'],
            maxstep=s['maxstep_A']
        )
        ok = bool(opt.run(fmax=fmax, steps=s['steps']))
        stages[label] = {
            'converged': ok,
            'steps': int(opt.nsteps),
            'fmax': float(np.linalg.norm(neb.get_forces(), axis=1).max())
        }
    energies = [float(a.get_potential_energy()) for a in images]
    save(P, 'neb_band.extxyz', images)
    res.update(neb=stages, band_energies_eV=energies)
    rows = []
    for idx in sorted(range(1, len(images) - 1), key=lambda i: (-energies[i], i)):
        probe = images[idx]
        direction = find_mic(
            images[idx + 1].positions - images[idx - 1].positions,
            probe.cell,
            probe.pbc
        )[0]
        if mic_collision_pairs(from_ase(probe), P.ads, TS_SETTINGS['collision_scale']):
            continue
        seed = probe.copy()
        seed.info = {'adsorbate_indices': P.ads}
        row, _ = validate(
            P,
            seed,
            direction,
            f'direct_img{idx}',
            p,
            ops,
            sperm,
            e_is,
            e_fs
        )
        row['image'] = idx
        rows.append(row)
        if p.get('mode'):
            p['_rt']['stop'] = _after_candidate(P, p, 'direct', row)
        break
    res['status'] = rows[0]['verdict'] if rows else 'no_valid_image'
    return (res, rows)


def scan_route(P, p, ops, sperm, e_is, e_fs):
    from ase.constraints import FixAtoms, FixBondLengths
    from ase.optimize import BFGS
    from ase.geometry import find_mic
    from chavis.search.planning import scan_targets, method2_seeds
    from chavis.search.automation_stages import tangent
    from chavis.inputs.is_from_fs import fragment
    s = TS_SETTINGS['scan']
    a0, b0 = P.pair.get('bond_cut') or _cut_bond(P)
    ref = {tuple(sorted(e)) for e in P.sp.ref}
    fa, fb = (fragment(ref, a0, (a0, b0)), fragment(ref, b0, (a0, b0)))
    anchor, lead, moving = (a0, b0, fb) if len(fb) <= len(fa) else (b0, a0, fa)
    d_is = float(P.IS.get_distance(anchor, lead, mic=True))
    cur = P.FS.copy()
    cur.info = dict(P.FS.info)
    pos = cur.positions
    pos[lead] = pos[anchor] + find_mic(pos[lead] - pos[anchor], cur.cell, cur.pbc)[0]
    for j in moving:
        if j != lead:
            pos[j] = pos[lead] + find_mic(pos[j] - pos[lead], cur.cell, cur.pbc)[0]
    cur.positions = pos
    d_fs = float(cur.get_distance(anchor, lead, mic=True))
    res = {
        'route': 'scan',
        'anchor': anchor,
        'lead': lead,
        'moving': moving,
        'd_IS_A': d_is,
        'd_FS_A': d_fs
    }
    if d_fs <= d_is:
        res['status'] = 'fs_bond_not_longer_than_is'
        return (res, [])
    frames, pts = ([], [])
    for k, target in enumerate(scan_targets(d_fs, d_is)):
        cur.set_constraint()
        v = cur.get_distance(anchor, lead, mic=True, vector=True)
        cur.positions[moving] += (target - np.linalg.norm(v)) * v / np.linalg.norm(v)
        cur.set_constraint([
            FixAtoms(indices=P.fixed),
            FixBondLengths([[anchor, lead]])
        ])
        cur.calc = P.calc()
        opt = BFGS(cur, maxstep=s['maxstep_A'], logfile=None)
        ok = bool(opt.run(fmax=s['fmax'], steps=s['steps']))
        e = float(cur.get_potential_energy())
        fr = cur.copy()
        fr.set_constraint(FixAtoms(indices=P.fixed))
        fr.info = {'adsorbate_indices': P.ads}
        frames.append(fr)
        pts.append({
            'k': k,
            'd_target_A': round(target, 4),
            'bond_A': round(float(cur.get_distance(anchor, lead, mic=True)), 4),
            'E_rel_IS_eV': e - e_is,
            'converged': ok,
            'steps': int(opt.nsteps)
        })
    save(P, 'scan_points.extxyz', frames)
    sel = method2_seeds(
        [x['E_rel_IS_eV'] for x in pts],
        [x['bond_A'] for x in pts],
        d_is
    )
    res.update(points=pts, selection=sel)
    rows = []
    if p.get('mode'):
        for stage, idxs in scan_candidate_order(sel):
            for idx in idxs:
                row, _ = validate(
                    P,
                    frames[idx],
                    tangent(frames, idx),
                    f'scan_pt{idx:02d}',
                    p,
                    ops,
                    sperm,
                    e_is,
                    e_fs
                )
                row.update(stage=stage, point=idx)
                rows.append(row)
                if _after_candidate(P, p, 'scan', row):
                    p['_rt']['stop'] = True
                    break
            if p['_rt']['stop']:
                break
        res['status'] = 'target_ts' if any((r['verdict'] == 'target_ts' for r in rows)) else 'no_target_ts'
        return (res, rows)
    stop = {
        'target_ts',
        'reaction_ts_other_endpoints'
    } if p.get('light') else {'target_ts'}
    for stage in ('stage_a',) if p.get('light') else ('stage_a', 'stage_b'):
        for idx in sel[stage]:
            row, _ = validate(
                P,
                frames[idx],
                tangent(frames, idx),
                f'scan_pt{idx:02d}',
                p,
                ops,
                sperm,
                e_is,
                e_fs
            )
            row.update(stage=stage, point=idx)
            rows.append(row)
            if row['verdict'] in stop:
                break
        if any((r['verdict'] in stop for r in rows)):
            break
    res['status'] = 'target_ts' if any((r['verdict'] == 'target_ts' for r in rows)) else 'no_target_ts'
    return (res, rows)


def _cut_bond(P):
    from chavis.judgment.structure_labels import adsorbate_graph
    gi = {tuple(sorted(e)) for e in adsorbate_graph(P.IS, P.ads, P.sp.rl)}
    gf = {tuple(sorted(e)) for e in adsorbate_graph(P.FS, P.ads, P.sp.rl)}
    lost = sorted(gi - gf)
    if len(lost) != 1:
        raise ValueError(f'expected one broken bond between IS and FS, got {lost}')
    return lost[0]


def run_pair(sp, pair, reaction, params, out, log=print):
    from chavis.surface.symmetry import surface_operations
    from chavis.inputs.pairs import slab_permutations
    p = {**clean_ts_params(params), 'light': bool(params.get('light'))}
    p['mode'] = check_mode(params.get('mode'), p['light'])
    P = Pair(sp, pair, reaction, p['ts_fix_layers'], out)
    ops = surface_operations(sp.slab)
    sperm = slab_permutations(
        sp.clean,
        sp.slab.get_chemical_symbols(),
        ops,
        sp.slab.cell.array
    )
    from chavis.surface.metal_mode import parse
    sp.ensure_calc()
    t0 = time.monotonic()
    c0 = sp.factory.calls
    result = {
        'reaction': reaction['id'],
        'equation': reaction['equation'],
        'route_plan': pair['route'],
        'effective_route_plan': p['ts_route_override'] or pair['route'],
        'pair_group': pair.get('pair_group'),
        'paired_is': pair['paired_is'],
        'fs_source': pair['fs_source'],
        'params': p,
        'settings': TS_SETTINGS,
        'fixed_layers': P.fix_layers,
        'fixed_atoms_snapped_A': P.fixed_snap_A
    }
    if p['mode']:
        result.update(
            mode=p['mode'],
            fast_stop_share=FAST_STOP_SHARE,
            judge_mode_min_share=0.01,
            candidate_order=('scan_stage_a_peak_outward>scan_stage_b>direct scan_then_direct' if p['ts_route_override'] == 'scan_then_direct' else CANDIDATE_ORDER)
        )
    endpoints = {}
    if parse(pair['fixed_layers']) != parse(p['ts_fix_layers']):
        P.IS, endpoints['IS'] = P.relax(P.IS, 'IS_rerelax')
        P.FS, endpoints['FS'] = P.relax(P.FS, 'FS_rerelax')
    e_is, e_fs = (P.energy(P.IS), P.energy(P.FS))
    save(P, 'IS.extxyz', P.IS)
    save(P, 'FS.extxyz', P.FS)
    result.update(E_IS_eV=e_is, E_FS_eV=e_fs, endpoint_rerelax=endpoints or None)
    routes = []
    if p['mode']:
        from chavis.judgment.judge_v3 import pair_context
        ctx = pair_context(
            sp,
            P.out,
            res={
                'reaction': reaction['id'],
                'fixed_layers': P.fix_layers,
                'E_IS_eV': e_is,
                'E_FS_eV': e_fs
            }
        )
        if ctx is None:
            raise RuntimeError('IS/FS not saved before the candidates; cannot judge in mode runs')
        p['_rt'] = {
            'ctx': ctx,
            'order': 0,
            'c0': c0,
            't0': t0,
            'fast_stop': None,
            'stop': False
        }
        old = P.out / 'candidates.jsonl'
        if old.exists():
            old.rename(P.out / f"candidates.partial_{time.strftime('%Y%m%dT%H%M%S')}.jsonl")
    done = {'target_ts', 'reaction_ts_other_endpoints'} if p['light'] else {'target_ts'}
    if p['ts_route_override'] == 'scan_then_direct':
        # scan_then_direct: one rule for every reaction = bond-length scan seeds first,
        # CI-NEB seed only if the scan did not settle the pair (mode runs: no within-pair stop; plain runs: no target TS).
        r, rows = scan_route(P, p, ops, sperm, e_is, e_fs)
        routes.append({**r, 'seeds': rows})
        log(f"    scan: {r['status']} ({len(rows)} dimer seeds)")
        need_direct = (not p['_rt']['stop']) if p['mode'] else (not any(row.get('verdict') in done for row in rows))
        if need_direct:
            r, rows = direct_route(P, p, ops, sperm, e_is, e_fs)
            routes.append({**r, 'seeds': rows})
            log(f"    direct: {r['status']}")
    if p['ts_route_override'] == 'direct_only' or (p['ts_route_override'] is None and pair['route'] == 'direct_then_scan'):
        r, rows = direct_route(P, p, ops, sperm, e_is, e_fs)
        routes.append({**r, 'seeds': rows})
        log(f"    direct: {r['status']}")
    if p['mode']:
        need_scan = not p['_rt']['stop']
    else:
        need_scan = pair['route'] == 'scan_first' or (routes and routes[-1]['status'] not in done)
    if p['ts_route_override'] in ('direct_only', 'scan_then_direct'):
        need_scan = False                                   # scan_then_direct: the scan already ran above
    elif p['ts_route_override'] == 'scan_only':
        need_scan = True
    if need_scan:
        r, rows = scan_route(P, p, ops, sperm, e_is, e_fs)
        routes.append({**r, 'seeds': rows})
        log(f"    scan: {r['status']} ({len(rows)} dimer seeds)")
    hits = [s for r in routes for s in r['seeds'] if s['verdict'] == 'target_ts']
    other = [s for r in routes for s in r['seeds'] if s['verdict'] == 'reaction_ts_other_endpoints']
    result.update(
        routes=routes,
        target_found=bool(hits),
        barrier_forward_eV=min((h['barrier_forward_eV'] for h in hits), default=None),
        reaction_ts_other_endpoints=bool(other),
        barrier_other_endpoints_eV=min(
            (h['barrier_forward_eV'] for h in other),
            default=None
        ),
        model_calls=sp.factory.calls - c0,
        seconds=round(time.monotonic() - t0, 1)
    )
    if p['mode']:
        rt = p.pop('_rt')
        result['fast_stop'] = rt['fast_stop']
        result['n_candidates'] = rt['order']
        result['params'] = {k: v for k, v in p.items()}
    (P.out / 'result.json').write_text(json.dumps(result, indent=1, default=float))
    return result
