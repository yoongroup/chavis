"""Scientific steps with explicit, in-process dimer continuation.

The route_v2 module provides the shared source of all
scientific settings and validation helpers. A pending session is not a result
or a disk restart: its live ASE objects must survive until advance or close.
"""
from contextlib import ExitStack
import time

import numpy as np

from chavis.search import route_v2 as R


def build_scan(P, p, e_is):
    """Return (route metadata, frames), without validating candidates."""
    from ase.constraints import FixAtoms, FixBondLengths
    from ase.optimize import BFGS
    from ase.geometry import find_mic
    from chavis.search.planning import scan_targets, method2_seeds
    from chavis.inputs.is_from_fs import fragment

    s = R.TS_SETTINGS['scan']
    a0, b0 = P.pair.get('bond_cut') or R._cut_bond(P)
    ref = {tuple(sorted(e)) for e in P.sp.ref}
    fa, fb = fragment(ref, a0, (a0, b0)), fragment(ref, b0, (a0, b0))
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
    res = dict(route='scan', anchor=anchor, lead=lead, moving=moving,
               d_IS_A=d_is, d_FS_A=d_fs)
    if d_fs <= d_is:
        res['status'] = 'fs_bond_not_longer_than_is'
        return res, []
    frames, pts = [], []
    for k, target in enumerate(scan_targets(d_fs, d_is)):
        cur.set_constraint()
        v = cur.get_distance(anchor, lead, mic=True, vector=True)
        cur.positions[moving] += (target - np.linalg.norm(v)) * v / np.linalg.norm(v)
        cur.set_constraint([FixAtoms(indices=P.fixed), FixBondLengths([[anchor, lead]])])
        cur.calc = P.calc()
        opt = BFGS(cur, maxstep=s['maxstep_A'], logfile=None)
        ok = bool(opt.run(fmax=s['fmax'], steps=s['steps']))
        e = float(cur.get_potential_energy())
        fr = cur.copy()
        fr.set_constraint(FixAtoms(indices=P.fixed))
        fr.info = {'adsorbate_indices': P.ads}
        frames.append(fr)
        pts.append(dict(k=k, d_target_A=round(target, 4),
                        bond_A=round(float(cur.get_distance(anchor, lead, mic=True)), 4),
                        E_rel_IS_eV=e - e_is, converged=ok, steps=int(opt.nsteps)))
    R.save(P, 'scan_points.extxyz', frames)
    res.update(points=pts, selection=method2_seeds(
        [x['E_rel_IS_eV'] for x in pts], [x['bond_A'] for x in pts], d_is))
    return res, frames


def build_neb_seed(P, p):
    """Return (direct-route metadata, one eligible seed or None).

    seed_info contains label, seed (Atoms), direction, and image. It is not a
    result row. The image choice, collision checks and two FIRE stages are
    exactly those of route_v2.direct_route.
    """
    from ase.mep import NEB
    from ase.optimize import FIRE
    from ase.geometry import find_mic
    from chavis.search.mic_idpp import prepare_idpp, mic_collision_pairs, PathPreparationError
    from chavis.search.geometry import to_ase, from_ase

    s = R.TS_SETTINGS['neb']
    res = {'route': 'direct'}
    try:
        frames, diag = prepare_idpp(R._structure(P.IS, P.fixed, P.ads),
                                    R._structure(P.FS, P.fixed, P.ads), R.TS_SETTINGS['images'])
    except PathPreparationError as exc:
        res.update(status='idpp_failure', reason=str(exc))
        return res, None
    res['idpp'] = diag.get('idpp')
    images = [to_ase(f) for f in frames]
    for a in images:
        a.calc = P.calc()
    neb = NEB(images, climb=False, method=s['method'])
    stages = {}
    for label, fmax, climb in (('warm', s['warm_fmax'], False), ('ci', s['ci_fmax'], True)):
        neb.climb = climb
        opt = FIRE(neb, logfile=str(P.out / f'neb_{label}.log'), dt=s['dt'], maxstep=s['maxstep_A'])
        ok = bool(opt.run(fmax=fmax, steps=s['steps']))
        stages[label] = dict(converged=ok, steps=int(opt.nsteps),
                             fmax=float(np.linalg.norm(neb.get_forces(), axis=1).max()))
    energies = [float(a.get_potential_energy()) for a in images]
    R.save(P, 'neb_band.extxyz', images)
    res.update(neb=stages, band_energies_eV=energies)
    for idx in sorted(range(1, len(images) - 1), key=lambda i: (-energies[i], i)):
        probe = images[idx]
        direction = find_mic(images[idx + 1].positions - images[idx - 1].positions,
                             probe.cell, probe.pbc)[0]
        if mic_collision_pairs(from_ase(probe), P.ads, R.TS_SETTINGS['collision_scale']):
            continue
        seed = probe.copy()
        seed.info = {'adsorbate_indices': P.ads}
        return res, dict(label=f'direct_img{idx}', seed=seed, direction=direction, image=idx)
    res['status'] = 'no_valid_image'
    return res, None


class DimerSession:
    """Live ASE dimer state; advance takes a cumulative translation-step limit.

    Construction is lazy so every model call belongs to an advance segment.
    The first limit is normally 300 and the final limit is 1500. Pausing does
    not close objects, evaluate extra forces, or serialize a partial TS.
    """

    def __init__(self, P, seed, direction, label, max_steps=1500):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        self.P, self.seed, self.direction, self.label = P, seed.copy(), np.array(direction, float), label
        self.seed.info = dict(seed.info)
        self.max_steps = max_steps
        self.state = 'pending'
        self.ts = self.control = self.mma = self.opt = None
        self.dimer = None
        self.calls_total, self.seconds_total = 0, 0.0
        self._stack = ExitStack()
        self._opened = self._closed = False

    @property
    def nsteps(self):
        return int(self.opt.nsteps) if self.opt is not None else 0

    def _open(self):
        from ase.constraints import FixAtoms
        from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
        self._opened = True
        self.ts = self.seed.copy()
        self.ts.info = dict(self.seed.info)
        self.ts.set_constraint(FixAtoms(indices=self.P.fixed))
        self.ts.calc = self.P.calc()
        v = self.direction.copy()
        v[self.P.fixed] = 0
        if np.linalg.norm(v) < 1e-12:
            self.ts = None
            self.state = 'failed'
            self.dimer = dict(converged=False, reason='zero_direction')
            return
        v /= np.linalg.norm(v)
        self.control = self._stack.enter_context(DimerControl(
            logfile=str(self.P.out / f'{self.label}_control.log'), eigenmode_logfile=None,
            initial_eigenmode_method='displacement', displacement_method='vector',
            mask=[i in set(self.P.free) for i in range(len(self.ts))]))
        self.mma = MinModeAtoms(self.ts, control=self.control, eigenmodes=[v])
        self.opt = self._stack.enter_context(MinModeTranslate(
            self.mma, logfile=str(self.P.out / f'{self.label}.log')))

    def advance(self, total_steps=300):
        if type(total_steps) is not int or not 1 <= total_steps <= self.max_steps:
            raise ValueError('total_steps must be an integer between 1 and max_steps')
        if self.state != 'pending':
            return self._result(0, 0.0)
        if self._closed:
            raise RuntimeError('Cannot resume a closed pending dimer session')
        if total_steps <= self.nsteps:
            raise ValueError('Continuation limit must exceed the current cumulative steps')
        c0, t0 = self.P.sp.factory.calls, time.monotonic()
        try:
            if not self._opened:
                self._open()
            if self.state == 'pending':
                try:
                    ok = bool(self.opt.run(fmax=R.TS_SETTINGS['dimer_fmax'],
                                           steps=total_steps - self.nsteps))
                    err = None
                except Exception as exc:
                    ok, err = False, f'{type(exc).__name__}: {exc}'
                pending = not ok and err is None and self.nsteps < self.max_steps
                if pending:
                    # ASE has just checked the gradient and stored the real
                    # forces/energy at this geometry. Reading these fields adds
                    # no model invocation or eigenmode rotation at a pause.
                    curv = float(self.mma.get_curvature())
                    fmax = float(np.linalg.norm(self.mma.forces0, axis=1).max())
                    energy = float(self.mma.energy0)
                    conv = False
                else:
                    curv = float(self.mma.get_curvature()) if err is None else None
                    fmax = float(np.linalg.norm(self.ts.get_forces(), axis=1).max()) if err is None else None
                    conv = bool(ok and fmax is not None and fmax <= R.TS_SETTINGS['dimer_fmax']
                                and curv is not None and curv < 0)
                    R.save(self.P, f'{self.label}_ts.extxyz', self.ts)
                    energy = float(self.ts.get_potential_energy()) if err is None else None
                    self.state = 'converged' if conv else 'failed'
                self.dimer = dict(converged=conv, optimizer_converged=ok, error=err,
                                  steps=self.nsteps, fmax=fmax, curvature=curv, energy_eV=energy)
            if self.state != 'pending':
                self.close()
        except BaseException:
            self.close()
            raise
        finally:
            calls, seconds = self.P.sp.factory.calls - c0, time.monotonic() - t0
            self.calls_total += calls
            self.seconds_total += seconds
        return self._result(calls, seconds)

    def _result(self, calls, seconds):
        return dict(state=self.state, ts=self.ts, dimer=dict(self.dimer or {}),
                    calls=calls, seconds=seconds, calls_total=self.calls_total,
                    seconds_total=self.seconds_total)

    def close(self):
        self._stack.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def validate_converged_ts(P, ts, dimer, label, p, ops, sperm, e_is, e_fs,
                          cost=None, ctx=None, route='scan'):
    """Finish route validation and judge_seed without another dimer run.

    Also handles final nonconvergence/zero direction, preserving the route
    rejected row and skipped stage costs. cost['dimer'] belongs to the caller
    and must contain the sum of all advance segments.
    """
    from chavis.judgment.judge_v3 import geometry, judge_seed, pair_context, MODE_MIN_SHARE, PairJudgmentCache
    cost = {} if cost is None else cost
    row = {'seed': label, 'dimer': dict(dimer), 'cost': cost}
    cache = getattr(P, '_judge_v3_cache', None)
    if cache is None:
        cache = P._judge_v3_cache = PairJudgmentCache(str(P.out.resolve()))
    if not dimer['converged']:
        cost.update(freq=R._skipped('dimer_not_converged'), conn=R._skipped('dimer_not_converged'))
        row['verdict'] = 'dimer_not_converged'
    else:
        row.update(barrier_forward_eV=dimer['energy_eV'] - e_is,
                   barrier_reverse_eV=dimer['energy_eV'] - e_fs)
        cached = cache.reuse(ts, row, route, P.out)
        if cached is not None:
            return cached
        geo = geometry(P.sp, ts, P.IS, [i for i in P.free if i < P.sp.n])
        row['geometry_precheck'] = geo
        if not geo['ok']:
            cost.update(freq=R._skipped(f"geometry_{geo['reason']}"),
                        conn=R._skipped(f"geometry_{geo['reason']}"))
            row['verdict'] = 'geometry_rejected'
        else:
            with R._Step(P, cost, 'freq'):
                mode, fr = R.frequencies(P, ts, label)
            row['frequencies'] = fr
            if mode is None:
                cost['conn'] = R._skipped(f"imaginary_{fr['imaginary_count']}")
                row['verdict'] = f"imaginary_{fr['imaginary_count']}"
            else:
                chem = {}
                with R._Step(P, cost, 'conn'):
                    connected, graphs, conn = R.connect(P, ts, mode, label, ops, sperm,
                                                        chemistry_out=chem)
                row['connection'] = conn
                row['reaction_graph_chemistry'] = chem or None
                row['reaction_mode'] = R.reaction_mode(
                    P, ts, mode, permutation=chem.get('permutation') if chem.get('ok') else None)
                row['connects_reaction_graphs'] = bool(graphs)
                along = row['reaction_mode']['along_bond']
                row['verdict'] = ('mode_not_along_reaction_bond' if (connected or graphs) and not along
                                  else 'target_ts' if connected else 'reaction_ts_other_endpoints'
                                  if graphs else 'connection_unresolved')
    if ctx is None:
        ctx = pair_context(P.sp, P.out, res=dict(reaction=P.reaction['id'], fixed_layers=P.fix_layers,
                                               E_IS_eV=e_is, E_FS_eV=e_fs))
    if ctx is None:
        raise RuntimeError('IS/FS not saved before candidate judgment')
    judged, extra = judge_seed(P.sp, P.out, route, row, ctx, MODE_MIN_SHARE)
    share = extra['share_raw']
    row['judged'] = dict(validated=extra['validated'], success=bool(judged.get('success')),
                         share_raw=share, failed_gate=judged.get('failed_gate'),
                         barrier_from_lower_reactant_eV=judged.get('barrier_from_lower_reactant_eV'),
                         fast_stop_candidate=bool(extra['validated'] and share is not None
                                                  and np.isfinite(share) and share >= MODE_MIN_SHARE))
    row['judged_full'] = judged
    cache.remember(ts, row, judged, extra, route, P.out)
    return row
