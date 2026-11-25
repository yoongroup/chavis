"""Round-robin scheduler and its unchanged-physics CHAVIS adapter."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_checkpoint(path, obj):
    """Publish a complete checkpoint atomically, with the data flushed first."""
    path, temporary = Path(path), None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(obj, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@dataclass
class PairState:
    pair: dict
    form: str
    key: str
    prepared: bool = False
    scanned: bool = False
    seeds: list = field(default_factory=list)
    cursor: int = 0
    pending: list = field(default_factory=list)
    cancelled: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    neb_built: bool = False
    neb_seed: object = None
    neb_started: bool = False
    scan_broken: bool = False
    complete: bool = False
    completion_reason: object = None
    success: object = None
    active_candidate: object = None
    deferrals: int = 0
    eligible_sweep: int = 0
    data: dict = field(default_factory=dict)


class Scheduler:
    """Backend operations own the physics; this class is CPU-testable with scripted outcomes."""
    def __init__(self, groups, backend, settings, event=None):
        self.groups = [[PairState(p, g['form'], f"pair_{p['pair_group']:03d}")
                        for p in g['pairs']] for g in groups]
        self.states = [s for g in self.groups for s in g]
        if len({s.key for s in self.states}) != len(self.states):
            raise ValueError('duplicate representative pair_group')
        self.backend, self.settings = backend, settings
        self.search_scope = settings.get('search_scope', 'first_success')
        if self.search_scope not in ('first_success', 'all_pairs'):
            raise ValueError('search_scope must be first_success or all_pairs')
        self.event = event or (lambda **kw: None)
        self.sweep = 0
        self.stop = None
        self.first_success = None
        self.final_order = 0

    def _defer(self, state, reasons):
        if not reasons or state.deferrals >= self.settings['max_deferrals_per_pair']:
            return False
        state.deferrals += 1
        state.eligible_sweep = self.sweep + 1
        group = next(g for g in self.groups if any(x is state for x in g))
        group.remove(state)
        group.append(state)
        self.event(kind='defer', pair=state.key, reasons=reasons, sweep=self.sweep)
        return True

    def _finish(self, state, candidate, outcome):
        from chavis.judgment.judge_v3 import should_stop
        if outcome['state'] == 'pending':
            state.pending.append(candidate)
            self.event(kind='pending', pair=state.key, seed=candidate['label'], sweep=self.sweep)
            return
        row = self.backend.finalize(state, candidate, outcome)
        self.final_order += 1
        row['reaction_order'] = self.final_order
        state.rows.append(row)
        self.backend.record_row(state, row)
        judged = row.get('judged') or {}
        share = judged.get('share_raw')
        stop_eligible = should_stop(judged.get('validated'), share)
        self.event(kind='candidate_final', pair=state.key, seed=candidate['label'],
                   validated=judged.get('validated', False), stop_eligible=stop_eligible, sweep=self.sweep)
        if stop_eligible:
            success = {'reason': 'validated_share', 'pair': state.key, 'seed': candidate['label']}
            state.success = dict(success)
            if self.first_success is None:
                self.first_success = dict(success)
                record = getattr(self.backend, 'record_first_success', None)
                if record is not None:
                    record(state, candidate, row)
            if self.search_scope == 'first_success':
                self.stop = success
            else:
                # A successful pair is terminal even if it still owns untried
                # scan seeds or suspended dimers. Preserve their spent costs.
                close_pair = getattr(self.backend, 'close_pair', None)
                if close_pair is not None:
                    close_pair(state)
                state.cancelled.extend(state.pending)
                state.pending.clear()
                state.complete = True
                state.completion_reason = 'validated_share'
                self.event(kind='pair_complete', pair=state.key, sweep=self.sweep)

    def _start(self, state, candidate):
        self.event(kind='candidate_start', pair=state.key, seed=candidate['label'], sweep=self.sweep)
        state.active_candidate = candidate
        outcome = self.backend.start(state, candidate, self.settings['dimer_screen_steps'])
        self._finish(state, candidate, outcome)
        state.active_candidate = None

    def _scan_final(self, state):
        return (state.scanned and state.cursor == len(state.seeds)
                and not any(c['route'] == 'scan' for c in state.pending))

    def _fallback(self, state, budget):
        # Pair-local eligibility: no untried scan seed and no pending scan dimer.
        if state.complete or not self._scan_final(state) or state.neb_started or self.stop:
            return budget
        if budget <= 0:
            return budget
        if not state.neb_built:
            state.neb_seed = self.backend.neb(state)
            state.neb_built = True
        state.neb_started = True
        if state.neb_seed is not None:
            self._start(state, state.neb_seed)
            budget -= 1
        return budget

    def _early_neb(self, state):
        # Early NEB: NEB on this pair after its k-th failed (finalized, not stop-eligible) candidate,
        # k = neb_after_failures_broken for a peakless (broken) scan, else neb_after_failures; the scan then continues.
        if state.neb_started or self.stop is not None or not state.scanned or state.complete:
            return False
        # Settings without these two keys leave the rule off; load_settings supplies 3 and 1.
        limit = self.settings.get('neb_after_failures_broken' if state.scan_broken else 'neb_after_failures', 0)
        if limit <= 0 or len(state.rows) < limit:      # before the NEB starts, every finalized row is a scan candidate
            return False
        if not state.neb_built:
            state.neb_seed = self.backend.neb(state)
            state.neb_built = True
        state.neb_started = True
        self.event(kind='early_neb', pair=state.key, broken=state.scan_broken, failures=len(state.rows), sweep=self.sweep)
        if state.neb_seed is not None:
            self._start(state, state.neb_seed)
        return True

    def _complete(self, state):
        if not state.complete and self._scan_final(state) and state.neb_started and not state.pending:
            state.complete = True
            state.completion_reason = 'validated_share' if state.success else 'candidates_exhausted'
            self.event(kind='pair_complete', pair=state.key, sweep=self.sweep)

    def _visit(self, state, short):
        self.event(kind='visit', pair=state.key, phase='screen' if short else 'resume', sweep=self.sweep)
        if not state.prepared:
            self.backend.prepare(state)
            state.prepared = True
        if not state.scanned:
            state.seeds, shape = self.backend.scan(state)
            state.scanned = True
            state.scan_broken = bool(shape.get('broken'))
            if self._defer(state, list(shape.get('reasons', [])) if shape.get('broken') else []):
                return
        budget = self.settings['seeds_per_visit'] if short else self.settings['resumes_per_visit']
        if short:
            while state.cursor < len(state.seeds) and budget > 0 and self.stop is None and not state.complete:
                candidate = state.seeds[state.cursor]
                state.cursor += 1
                self._start(state, candidate)
                budget -= 1
                if self.stop is None and self._early_neb(state):
                    budget -= 1
        else:
            # Snapshot pending queue: a CI seed started later in this visit is not resumed immediately.
            existing = state.pending[:budget]
            for candidate in existing:
                state.pending.remove(candidate)
                state.active_candidate = candidate
                outcome = self.backend.resume(state, candidate, self.settings['dimer_max_steps'])
                self._finish(state, candidate, outcome)
                state.active_candidate = None
                budget -= 1
                if self.stop is not None or state.complete:
                    break
                if self._early_neb(state):
                    budget -= 1
                    if self.stop is not None or state.complete:
                        break
        if self.stop is None and not state.complete:
            self._fallback(state, budget)
            self._complete(state)

    def run(self):
        try:
            while self.stop is None and any(not s.complete for s in self.states):
                self.sweep += 1
                short = any(not s.scanned or s.cursor < len(s.seeds) for s in self.states if not s.complete)
                order = [g[i] for i in range(max(map(len, self.groups), default=0))
                         for g in self.groups if i < len(g)]
                progressed = False
                for state in order:
                    if state.complete or state.eligible_sweep > self.sweep:
                        continue
                    self._visit(state, short)
                    progressed = True
                    if self.stop is not None:
                        break
                if not progressed:
                    raise RuntimeError('scheduler made no progress')
            return self.stop or {'reason': ('all_pairs_complete' if self.search_scope == 'all_pairs'
                                           else 'pool_exhausted'), 'pair': None, 'seed': None}
        finally:
            self.backend.close(self.states)


class ScientificBackend:
    def __init__(self, species, reaction, params, settings, out, ledger):
        from chavis.search import route_v2 as R
        from chavis.surface.symmetry import surface_operations
        from chavis.inputs.pairs import slab_permutations
        self.R, self.sp, self.reaction = R, species, reaction
        self.params = {**R.clean_ts_params(params), 'mode': 'fast', 'light': False,
                       'dimer_max_steps': settings['dimer_max_steps']}
        self.settings, self.out, self.ledger = settings, Path(out), ledger
        self.ops = surface_operations(species.slab)
        self.sperm = slab_permutations(species.clean, species.slab.get_chemical_symbols(),
                                       self.ops, species.slab.cell.array)
        self.sessions = []
        self.first_success_checkpoint = None

    def prepare(self, state):
        from chavis.surface.metal_mode import parse
        from chavis.judgment.judge_v3 import pair_context
        R = self.R
        with self.ledger.segment('pair_prepare', state.key):
            P = R.Pair(self.sp, state.pair, self.reaction, self.params['ts_fix_layers'], self.out / state.key)
            endpoint_relax = {}
            if parse(state.pair['fixed_layers']) != parse(self.params['ts_fix_layers']):
                P.IS, endpoint_relax['IS'] = P.relax(P.IS, 'IS_rerelax')
                P.FS, endpoint_relax['FS'] = P.relax(P.FS, 'FS_rerelax')
            e_is, e_fs = P.energy(P.IS), P.energy(P.FS)
            R.save(P, 'IS.extxyz', P.IS)
            R.save(P, 'FS.extxyz', P.FS)
            ctx = pair_context(self.sp, P.out, {'reaction': self.reaction['id'],
                               'fixed_layers': P.fix_layers, 'E_IS_eV': e_is, 'E_FS_eV': e_fs})
            if ctx is None:
                raise RuntimeError('endpoint context unavailable')
            bond = P.pair.get('bond_cut') or R._cut_bond(P)
            distance = float(P.FS.get_distance(*bond, mic=True))
            state.data.update(P=P, e_is=e_is, e_fs=e_fs, ctx=ctx, bond=bond,
                              endpoint_relax=endpoint_relax, product_bond_A=distance)
        return distance

    def scan(self, state):
        from .steps import build_scan
        from .planning import classify_scan, order_seeds
        P = state.data['P']
        with self.ledger.segment('scan', state.key):
            meta, frames = build_scan(P, self.params, state.data['e_is'])
        points = meta.get('points', [])
        if points:
            shape = classify_scan(points, self.settings)
            old = [(stage, i) for stage, indices in self.R.scan_candidate_order(meta['selection']) for i in indices]
            ordered = order_seeds(points, [i for _, i in old],
                                  [P.IS.get_chemical_symbols()[i] for i in state.data['bond']],
                                  shape, self.settings)
            stages = {i: stage for stage, i in old}
            seeds = [{'route': 'scan', 'label': f'scan_pt{i:02d}', 'point': i,
                      'stage': stages[i]} for i in ordered['order']]
        else:
            if meta.get('status') != 'fs_bond_not_longer_than_is':
                raise ValueError('empty scan without an explicit legacy no-scan status')
            shape = {'broken': False, 'reasons': [], 'no_scan': meta.get('status')}
            ordered = {'order': [], 'prior_applied': False}
            seeds = []
        meta.update(shape=shape, ultra_seed_order=ordered)
        state.data.update(scan=meta, frames=frames)
        write_json(P.out / 'scan.json', meta)
        return seeds, shape

    def neb(self, state):
        from .steps import build_neb_seed
        with self.ledger.segment('neb', state.key):
            meta, seed = build_neb_seed(state.data['P'], self.params)
        state.data['direct'] = meta
        write_json(state.data['P'].out / 'neb.json', meta)
        if seed is not None:
            seed = dict(seed, route='direct')
        return seed

    def start(self, state, candidate, limit):
        from .steps import DimerSession
        from chavis.search.automation_stages import tangent
        if candidate['route'] == 'scan':
            frames, i = state.data['frames'], candidate['point']
            seed, direction = frames[i], tangent(frames, i)
        else:
            seed, direction = candidate['seed'], candidate['direction']
        with self.ledger.segment('dimer', state.key, candidate['label'], cumulative_limit=limit):
            session = DimerSession(state.data['P'], seed, direction, candidate['label'],
                                   max_steps=self.settings['dimer_max_steps'])
            candidate['_session'] = session
            self.sessions.append(session)
            return session.advance(limit)

    def resume(self, state, candidate, limit):
        with self.ledger.segment('dimer', state.key, candidate['label'], cumulative_limit=limit):
            return candidate['_session'].advance(limit)

    def finalize(self, state, candidate, outcome):
        from .steps import validate_converged_ts
        label = candidate['label']
        cost = {'dimer': dict(self.ledger.totals(pair=state.key, seed=label, stage='dimer'), status='ran')}
        with self.ledger.segment('validation', state.key, label):
            row = validate_converged_ts(state.data['P'], outcome.get('ts'), outcome['dimer'],
                      label, self.params, self.ops, self.sperm, state.data['e_is'], state.data['e_fs'],
                      cost=cost, ctx=state.data['ctx'], route=candidate['route'])
        row.update(route=candidate['route'], seed_id=f"{candidate['route']}:{label}",
                   checkpoint=self.ledger.totals(pair=state.key))
        for k in ('point', 'stage', 'image'):
            if k in candidate:
                row[k] = candidate[k]
        if self.settings.get('search_scope', 'first_success') == 'all_pairs':
            self._release_session(candidate)
        else:
            candidate['_session'].close()
        return row

    def record_row(self, state, row):
        with (state.data['P'].out / 'candidates.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()

    def record_first_success(self, state, candidate, row):
        """Freeze the global counter at validation, including other pair work.

        This ledger covers the TS reaction process. The campaign adds all
        already incurred preparation/attempt costs to its separate checkpoint.
        Pair totals must never be substituted for this global snapshot.
        """
        costs = self.ledger.summary()
        self.first_success_checkpoint = {
            'schema': 'chavis_ultra_first_success_v1',
            'scope': 'ts_reaction_process',
            'pair': state.key, 'seed': candidate['label'], 'route': candidate['route'],
            'reaction_order': row['reaction_order'],
            'share_raw': row['judged']['share_raw'],
            'ts_energy_eV': row['dimer']['energy_eV'],
            'model_calls': costs['total']['calls'], 'seconds': costs['elapsed_seconds'],
            'cost_ledger': costs,
            'accounting': 'All TS process calls through the first validated share >= 0.01; '
                          'includes model initialization and unfinished work on every pair. '
                          'Campaign preparation and previous attempts are accounted separately.'}
        if hasattr(self, 'attempt_id'):
            self.first_success_checkpoint.update(attempt_id=self.attempt_id,
                                                request_fingerprint=self.fingerprint,
                                                reference_is_energy_eV=self.reference_is_energy_eV,
                                                barrier_eV=row['dimer']['energy_eV']
                                                - self.reference_is_energy_eV)
        write_checkpoint(self.out / 'first_success_checkpoint.json', self.first_success_checkpoint)

    def close_pair(self, state):
        """Release suspended sessions as soon as their own pair has succeeded."""
        for candidate in state.pending:
            self._release_session(candidate)

    def _release_session(self, candidate):
        session = candidate['_session']
        session.close()
        del candidate['_session']
        self.sessions = [live for live in self.sessions if live is not session]

    def close(self, states):
        for session in self.sessions:
            session.close()

    def results(self, states, stop, reference_is_energy_eV, attempt_id, fingerprint,
                status='complete', error=None):
        valid = []
        for state in states:
            if not state.prepared:
                continue
            routes = []
            for name in ('scan', 'direct'):
                if name in state.data:
                    routes.append({**state.data[name], 'route': name,
                                   'seeds': [r for r in state.rows if r['route'] == name]})
            pending = [{'route': c['route'], 'seed': c['label'],
                        'status': ('pending_at_reaction_stop' if status == 'complete'
                                   else 'closed_at_interruption'),
                        'cost': self.ledger.totals(pair=state.key, seed=c['label'])}
                       for c in state.pending]
            cancelled = [{'route': c['route'], 'seed': c['label'],
                          'status': 'cancelled_after_pair_success',
                          'cost': self.ledger.totals(pair=state.key, seed=c['label'])}
                         for c in state.cancelled]
            interrupted = ([{'route': state.active_candidate['route'],
                             'seed': state.active_candidate['label'],
                             'status': 'interrupted_candidate',
                             'cost': self.ledger.totals(pair=state.key,
                                                       seed=state.active_candidate['label'])}]
                           if state.active_candidate is not None else [])
            result = {'reaction': self.reaction['id'], 'equation': self.reaction['equation'],
                      'mode': 'ultra-fast', 'pair_group': state.pair['pair_group'],
                      'paired_is': state.pair['paired_is'], 'fs_source': state.pair['fs_source'],
                      'form': state.form, 'params': self.params, 'ultra_settings': self.settings,
                      'structure_priority': state.pair.get('_ultra_structure'),
                      'effective_grade': state.pair.get('_ultra_grade'),
                      'grade_ignored_all_floating': state.pair.get('_ultra_grade_ignored_all_floating'),
                      'order_mode': self.settings['order_mode'],
                      'effective_priority': state.pair.get('_ultra_effective_priority'),
                      'demoted': state.pair.get('_ultra_demoted'),
                      'fixed_layers': state.data['P'].fix_layers,
                      'E_IS_eV': state.data['e_is'], 'E_FS_eV': state.data['e_fs'],
                      'routes': routes, 'pending': pending, 'cancelled': cancelled,
                      'interrupted_candidates': interrupted,
                      'complete': state.complete, 'completion_reason': state.completion_reason,
                      'pair_success': state.success, 'search_scope': self.settings['search_scope'],
                      'deferrals': state.deferrals, 'model_calls': self.ledger.totals(pair=state.key)['calls'],
                      'seconds': self.ledger.totals(pair=state.key)['seconds'],
                      'attempt_id': attempt_id, 'request_fingerprint': fingerprint,
                      'accounting': 'Active segments only; overlapping pair elapsed intervals are not summed.'}
            write_json(state.data['P'].out / 'result.json', result)
            for row in state.rows:
                if (row.get('judged') or {}).get('validated') is True:
                    energy = row['dimer']['energy_eV']
                    valid.append({'pair': state.key, 'seed': row['seed'], 'route': row['route'],
                                  'reaction_order': row['reaction_order'],
                                  'ts_energy_eV': energy,
                                  'barrier_eV': energy - reference_is_energy_eV,
                                  'share_raw': row['judged']['share_raw']})
        best = min(valid, key=lambda x: (x['ts_energy_eV'], x['pair'], x['seed'])) if valid else None
        first = min(valid, key=lambda x: x['reaction_order']) if valid else None
        costs = self.ledger.summary()
        checkpoint = self.first_success_checkpoint
        return {'schema': 'chavis_ultra_reaction_v2_2', 'status': status, 'mode': 'ultra-fast',
                'order_mode': self.settings['order_mode'],
                'search_scope': self.settings['search_scope'],
                'reaction': self.reaction['id'], 'equation': self.reaction['equation'],
                'validated': bool(valid), 'best_barrier_eV': best['barrier_eV'] if best else None,
                'best_ts_energy_eV': best['ts_energy_eV'] if best else None,
                'reference_is_energy_eV': reference_is_energy_eV,
                'barrier_reference': 'lowest full input reactant pool, including derived reactants',
                'first_validated': first['pair'] if first else None,
                'first_validated_candidate': first,
                'first_validated_barrier_eV': first['barrier_eV'] if first else None,
                'first_validated_ts_energy_eV': first['ts_energy_eV'] if first else None,
                'best_validated_candidate': best,
                'first_success_checkpoint': checkpoint,
                'first_validated_model_calls': checkpoint['model_calls'] if checkpoint else None,
                'stopping_pair': stop['pair'], 'stop_reason': stop['reason'], 'stop': stop,
                'n_pairs': len(states), 'pairs_run': sum(s.prepared for s in states),
                'validated_candidates': valid, 'pending_candidates': sum(len(s.pending) for s in states),
                'cancelled_candidates': sum(len(s.cancelled) for s in states),
                'interrupted_candidates': sum(s.active_candidate is not None for s in states),
                'pairs_complete': sum(s.complete for s in states),
                'pairs_succeeded': sum(s.success is not None for s in states),
                'pair_statuses': [{'pair': s.key, 'prepared': s.prepared, 'scanned': s.scanned,
                                   'complete': s.complete, 'completion_reason': s.completion_reason,
                                   'success': s.success, 'remaining_scan_seeds': len(s.seeds) - s.cursor,
                                   'pending_candidates': len(s.pending),
                                   'interrupted_candidate': (s.active_candidate['label']
                                                             if s.active_candidate else None),
                                   'cancelled_candidates': len(s.cancelled)} for s in states],
                'model_calls': costs['total']['calls'], 'seconds': costs['elapsed_seconds'],
                'cost_ledger': costs, 'attempt_id': attempt_id, 'request_fingerprint': fingerprint,
                'ultra_settings': self.settings, 'error': error,
                'continuation': 'reaction_boundary_fresh_attempt_only'}


def run_reaction(species, pairs, reaction, params, settings, metadata, out, ledger,
                 reference_is_energy_eV, attempt_id, fingerprint, log=print):
    from .planning import order_pairs, annotate_pair_structures, load_settings
    settings = load_settings(overrides=settings)
    groups = order_pairs(annotate_pair_structures(species, pairs, settings), metadata, settings)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    def event(**row):
        with (out / 'schedule.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        if row['kind'] in ('defer', 'candidate_final', 'pair_complete'):
            log(json.dumps(row))
    write_json(out / "structure_priority.json", {"schema": "endpoint_structure_priority_v2_2",
               "order_mode": settings['order_mode'], "groups": groups})
    backend = ScientificBackend(species, reaction, params, settings, out, ledger)
    backend.attempt_id, backend.fingerprint = attempt_id, fingerprint
    backend.reference_is_energy_eV = reference_is_energy_eV
    scheduler = Scheduler(groups, backend, settings, event)
    try:
        stop = scheduler.run()
    except BaseException as exc:
        status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        # Suspended optimizer sessions are deliberately not disk-resumable.
        # Save evidence and actual costs; the campaign may retry in a fresh
        # attempt and must retain this attempt's incurred costs.
        try:
            result = backend.results(scheduler.states,
                {'reason': status, 'pair': None, 'seed': None},
                reference_is_energy_eV, attempt_id, fingerprint, status=status,
                error=f'{type(exc).__name__}: {exc}')
            write_json(out / 'reaction_result.json', result)
            write_json(out / 'cost_ledger.json', ledger.summary())
        except BaseException as report_error:
            # Do not replace the scientific/infrastructure exception if a
            # failed filesystem or counter also prevents partial reporting.
            if hasattr(exc, 'add_note'):
                exc.add_note(f'Could not save partial reaction result: {report_error}')
        raise
    result = backend.results(scheduler.states, stop, reference_is_energy_eV, attempt_id, fingerprint)
    write_json(out / 'reaction_result.json', result)
    return result
