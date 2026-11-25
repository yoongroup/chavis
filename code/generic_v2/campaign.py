"""Durable reaction-boundary resume; a failed attempt is retained and never reused."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .common import ROOT, atomic_json, digest_json, environment, read_json, sha256
from .process import parent_death_options
from .surfaces import DEFAULT_SURFACE, inspect_cache, surface_definition
from reproduce.drivers.ultra_fast import Campaign as V2Checks, number


def cost_summary(attempt):
    """New calls of this attempt; imported preparation lives in campaign.imported_charges.

    A missing cost envelope is unknown, never silently counted as zero."""
    known, seconds, unknown, unknown_seconds = 0, 0.0, [], []
    surface_phases = {'eos': 0, 'slab': 0}
    for path in sorted((Path(attempt) / 'stage_requests').glob('*.json')):
        req = read_json(path)
        try:
            c = read_json(Path(attempt) / req['cost_file'])
            if not isinstance(c, dict):
                raise ValueError('cost envelope must be an object')
            calls = c['calls']
            if not number(calls) or calls < 0 or int(calls) != calls:
                raise ValueError('invalid calls')
            known += int(calls)
            if req.get('tool') == 'generic_v2.surface_stage':
                phases = c.get('phase_costs')
                if not isinstance(phases, dict) or set(phases) - set(surface_phases):
                    raise ValueError('missing or invalid surface phases')
                if c.get('error') is None and set(phases) != set(surface_phases):
                    raise ValueError('successful surface stage lacks EOS/slab phases')
                measured = {}
                for phase, row in phases.items():
                    value = row.get('calls') if isinstance(row, dict) else None
                    if not number(value) or value < 0 or int(value) != value:
                        raise ValueError('invalid surface phase cost')
                    measured[phase] = int(value)
                if sum(measured.values()) != calls:
                    raise ValueError('surface phase costs do not sum to total')
                for phase, value in measured.items():
                    surface_phases[phase] += value
            if number(c.get('seconds')) and c['seconds'] >= 0:
                seconds += c['seconds']
            else:
                unknown_seconds.append(req['key'])
        except (OSError, ValueError, KeyError):
            unknown.append(req['key'])
            unknown_seconds.append(req['key'])
    return {'known_calls': known,
            'known_stage_seconds': seconds, 'unknown_cost_stages': unknown,
            'known_surface_eos_calls': surface_phases['eos'], 'known_surface_slab_calls': surface_phases['slab'],
            'unknown_time_stages': unknown_seconds, 'calls_complete': not unknown, 'times_complete': not unknown_seconds}


class StageContext(V2Checks):
    """Reuse stage checks, with durable per-stage requests before spawning."""
    def __init__(self, path, request, reaction_ids, campaign):
        self.run = Path(path)
        self.run.mkdir(parents=True, exist_ok=False)
        (self.run / 'logs').mkdir()
        self.params = self.run / 'params.json'
        self.settings = self.run / 'ultra_settings.json'
        atomic_json(self.params, request['params'])
        atomic_json(self.settings, request['settings'])
        self.ledger = self.run / 'ledger.jsonl'
        self.run_id = campaign.state['run_id']
        self.reactions = {r['id']: r for r in request['cohort']['rows']}
        self.stage_records = []
        self.reaction_ids = reaction_ids
        self.campaign = campaign
        surface = request.get('surface', surface_definition())
        self.surf, self.surf_arg = surface['key'], surface['argument']

    def stage(self, key, tool, args, check, ultra=False):
        attempt = uuid.uuid4().hex
        cost = self.run / 'stage_costs' / (attempt + '.json')
        args = list(map(str, args))
        cmd = ([sys.executable, '-B', '-m', 'ultra_fast_v1.cli', *args, '--cost-file', str(cost)] if ultra else
               [sys.executable, '-B', str(ROOT / 'reproduce/drivers/stage.py'), str(cost), tool, *args])
        if tool == 'generic_v2.surface_stage':
            cmd = [sys.executable, '-B', '-m', tool, *args, '--cost-file', str(cost)]
        req = {'key': key, 'argv': cmd, 'tool': tool,
               'cost_file': str(cost.relative_to(self.run)), 'start_epoch': time.time()}
        atomic_json(self.run / 'stage_requests' / (attempt + '.json'), req)
        env = environment()
        env['CHAVIS_REACTION_IDS'] = json.dumps(self.reaction_ids)
        # No model lives in this driver. Exactly one stage child runs at a time.
        t0 = time.monotonic()
        inherited = parent_death_options()
        if os.name == 'posix' and 'CHAVIS_GENERAL_GPU_LOCK_FD' in env:
            inherited['pass_fds'] = (int(env['CHAVIS_GENERAL_GPU_LOCK_FD']),)
        with (self.run / 'logs' / (key + '.log')).open('ab') as log:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, **inherited).returncode
        c, error = {}, None
        try:
            c = read_json(cost)
            if not isinstance(c, dict):
                c = {}
                raise ValueError('cost envelope must be an object')
            error = c.get('error') or (f'process exit {rc}' if rc else None)
            if not number(c.get('calls')) or c['calls'] < 0 or int(c['calls']) != c['calls']:
                error = error or 'invalid model-call count'
            if not number(c.get('seconds')) or c['seconds'] < 0:
                error = error or 'invalid stage seconds'
            if error is None:
                check(c)
        except Exception as exc:
            error = repr(exc)
        rec = {**req, 'attempt': attempt, 'tool': tool, 'rc': rc, 'ok': error is None,
               'error': error, 'calls': c.get('calls'), 'seconds': c.get('seconds'),
               'subprocess_seconds': time.monotonic() - t0, 'end_epoch': time.time()}
        self.write(**rec)
        self.stage_records.append(rec)
        if error:
            raise RuntimeError(f'{key}: {error}')

    EXPECTED_RECORDS = 30

    def check_import(self, sp):
        """The comparison contract is the baseline's complete pool: exactly 30 records (placements 0..29), every one mapped
        onto the CHAVIS molecule. A partially converted pool must never pass as the same 30."""
        self.check_wf2(sp)
        doc = read_json(self.need(self.sp_dir(sp) / 'external_import.json', 'external import manifest'))
        n = self.EXPECTED_RECORDS
        indices = sorted(int(r['baseline_index']) for r in doc.get('records', []))
        problems = [name for name, ok in (
            ('n_records', doc.get('n_records') == n), ('baseline_placements', doc.get('baseline_placements') == n),
            ('n_mapped', doc.get('n_mapped') == n), ('n_mapping_failed', doc.get('n_mapping_failed') == 0),
            ('record_indices', indices == list(range(n)))) if not ok]
        if problems:
            raise RuntimeError(f'external pool of {sp} is not the complete baseline pool: {problems} '
                               f"(records {doc.get('n_records')}, placements {doc.get('baseline_placements')}, mapped {doc.get('n_mapped')})")
        if not number(doc.get('imported_reactant_calls')) or doc['imported_reactant_calls'] < 0 or int(doc['imported_reactant_calls']) != doc['imported_reactant_calls']:
            raise RuntimeError('external import manifest lacks a valid imported reactant call count')
        return doc


class Pipeline:
    def prepare_surface(self, campaign, path):
        ctx = StageContext(path, campaign.request, [], campaign)
        expected = ctx.run / 'expected_surface_cache.json'
        atomic_json(expected, campaign.request['surface_cache'])
        store, result = ctx.run / 'slab_store', ctx.run / 'surface_preparation.json'

        def check(cost):
            doc = read_json(ctx.need(result, 'surface preparation'))
            if doc['prepared_cache'] != inspect_cache(store, ctx.surf_arg):
                raise RuntimeError('Prepared slab cache does not match its recorded hashes')
            if sum(row['calls'] for row in cost['phase_costs'].values()) != cost['calls']:
                raise RuntimeError('Surface EOS/slab phase counts do not sum to the actual total')
        ctx.stage('prepare_surface', 'generic_v2.surface_stage',
                  ['--surface', ctx.surf_arg, '--source-store', ROOT / 'slab_store', '--store', store,
                   '--expected-cache', expected, '--out', result], check)
        return {'surface': campaign.request['surface'], 'store': str(store.relative_to(campaign.out)),
                'result_file': str(result.relative_to(ctx.run))}

    @staticmethod
    def surface_store(campaign):
        entry = campaign.state.get('surface_preparation')
        return campaign.out / entry['store'] if entry else ROOT / 'slab_store'

    def prepare(self, campaign, sp, path, rid=None):
        external = campaign.request.get('external_reactants')
        if external and rid is None:
            raise ValueError('External reactant records are prepared per reaction; rid is required')
        ids = [rid] if external else [r['id'] for r in campaign.request['cohort']['rows'] if r['parent_species'] == sp]
        ctx = StageContext(path, campaign.request, ids, campaign)
        payload = {'species': sp, 'reactions': ids, 'external': bool(external)}
        if external:
            # Same-pool comparison: the baseline's relaxed reactant records of this reaction and seed replace wf2.
            ctx.stage(f'import_{sp}_{rid}', 'import_reactants.py', ['--surface', ctx.surf_arg, '--species', sp,
                      '--reaction', rid, '--seed', campaign.request['seed'], '--source-seed-dir', external['seed_dir'],
                      '--out', ctx.run, '--store', ROOT / 'slab_store'], lambda c: ctx.check_import(sp))
            doc = read_json(ctx.sp_dir(sp) / 'external_import.json')
            charge = campaign.charge_import(rid, sp, doc)
            payload.update(imported_reactant_calls=charge['calls'], imported_charge_key=charge['key'], imported_from=doc['source_dir'],
                           records=doc['n_records'], records_mapped=doc['n_mapped'])
        else:
            azimuth = campaign.request['settings'].get('adsorption_azimuth_step_deg', 'auto')
            ctx.stage(f'wf2_{sp}', 'run_wf2.py', ['--surface', ctx.surf_arg, '--species', sp,
                      '--out', ctx.run, '--mode', 'fast', '--fix-layers', 'all', '--store', self.surface_store(campaign),
                      '--adsorption-azimuth-step-deg', azimuth], lambda c: ctx.check_wf2(sp))
        ctx.stage(f'select_{sp}', 'select_wf3.py', [ctx.run, '--params', ctx.params, '--rebuild'],
                  lambda c: ctx.check_select(sp))
        ctx.stage(f'wf3_{sp}', 'run_wf3.py', [ctx.run, '--params', ctx.params, '--surface', ctx.surf, '--species', sp],
                  lambda c: ctx.check_wf3(sp))
        rows = read_json(ctx.run / 'wf3_is.json')['surfaces'][ctx.surf]
        if set(rows) != set(ids):
            raise ValueError(f'Scoped wf3 reactions differ: expected {ids}, found {list(rows)}')
        return payload

    def reaction(self, campaign, rid, path, preparation):
        ctx = StageContext(path, campaign.request, [rid], campaign)
        sp = ctx.reactions[rid]['parent_species']
        # Physical copies permit wf4 derived reactants without changing frozen preparation.
        shutil.copytree(Path(preparation) / ctx.surf, ctx.run / ctx.surf)
        for name in ('wf3_is.json', 'wf3_selection.json'):
            shutil.copy2(Path(preparation) / name, ctx.run / name)
        ctx.stage(f'wf4_{rid}', 'run_wf4.py', [ctx.run, '--surface', ctx.surf, '--reaction', rid, '--params', ctx.params],
                  lambda c: ctx.need(ctx.sp_dir(sp) / f'fs_{rid}.jsonl', 'FS'))
        ctx.stage(f'pairs_{rid}', 'run_wf5_pairs.py', [ctx.run, '--surface', ctx.surf, '--reaction', rid, '--params', ctx.params],
                  lambda c: ctx.rep_pairs(rid))
        context = self.checkpoint_context(campaign, preparation, ctx.run, rid=rid)
        atomic_json(ctx.run / 'first_success_call_context.json', context)
        stage_error = None
        try:
            ctx.stage(f'ts_{rid}', 'ultra_fast_v1.cli', [ctx.run, '--params', ctx.params, '--settings', ctx.settings,
                      '--surface', ctx.surf, '--reaction', rid], lambda c: ctx.check_ultra(rid, c), ultra=True)
        except BaseException as exc:
            stage_error = exc
            raise
        finally:
            checkpoint_path = ctx.result_path(rid).parent / 'first_success_checkpoint.json'
            if checkpoint_path.is_file():
                try:
                    self.write_checkpoint(ctx.run, read_json(checkpoint_path), context)
                except Exception as checkpoint_error:
                    if stage_error is None:
                        raise
                    # A malformed/partial checkpoint must not hide the original
                    # interruption or stage error. Its cost is explicitly unknown.
                    if hasattr(stage_error, 'add_note'):
                        stage_error.add_note(f'First-success checkpoint unreadable: {checkpoint_error!r}')
                    try:
                        atomic_json(ctx.run / 'first_success_accounting_error.json', {
                            'checkpoint': str(checkpoint_path.relative_to(ctx.run)),
                            'error': repr(checkpoint_error), 'first_success_calls_complete': False})
                    except OSError:
                        pass
        result = read_json(ctx.result_path(rid))
        payload = {'result_file': str(ctx.result_path(rid).relative_to(ctx.run)),
                'validated': result.get('validated'), 'first_validated': result.get('first_validated'),
                'best_barrier_eV': result.get('best_barrier_eV'), 'ts_calls': result['model_calls']}
        if result.get('first_success_checkpoint'):
            payload['first_success_checkpoint'] = result['first_success_checkpoint']
            payload['first_success_accounting'] = self.write_checkpoint(
                ctx.run, result['first_success_checkpoint'], context)
        return payload

    @staticmethod
    def checkpoint_context(campaign, preparation, reaction, rid=None):
        prep, current = cost_summary(preparation), cost_summary(reaction)
        attempts = campaign.state['attempts']
        charges = campaign.state.get('imported_charges', {})
        if charges and rid is None:
            raise ValueError('A reaction ID is required to allocate imported preparation to a per-reaction first-success checkpoint')
        imported = sum(charge['calls'] for charge in charges.values())
        reaction_charges = {key: charge for key, charge in charges.items() if charge['reaction'] == rid}
        reaction_imported = sum(charge['calls'] for charge in reaction_charges.values())
        return {
            'reaction': rid, 'preparation_calls': prep['known_calls'] + reaction_imported,
            'preparation_new_model_calls': prep['known_calls'],
            'preparation_imported_calls': reaction_imported,
            'reaction_imported_charge_keys': sorted(reaction_charges),
            'campaign_imported_calls': imported,
            'reaction_pre_ts_calls': current['known_calls'],
            'per_reaction_pre_ts_calls': prep['known_calls'] + reaction_imported + current['known_calls'],
            'per_reaction_calls_complete': prep['calls_complete'] and current['calls_complete'],
            'campaign_pre_ts_calls': sum(a['known_calls'] for a in attempts) + imported + current['known_calls'],
            'campaign_calls_complete': all(a['calls_complete'] for a in attempts) and current['calls_complete'],
            'prior_abandoned_calls': sum(a['known_calls'] for a in attempts if a['status'] != 'complete'),
            'surface_preparation_calls': sum(a['known_calls'] for a in attempts if a['kind'] == 'surface'),
            'scope': 'Per-reaction paired analysis is the current attempt first success, not the earliest success '
                     'across retries; it charges full shared preparation plus the reaction imported charge once, '
                     'WF4/pair stages and global TS checkpoint. Campaign cumulative additionally includes earlier '
                     'reactions, all imported charges, surface preparation and abandoned attempts and is not a paired '
                     'per-reaction cost'}

    @staticmethod
    def write_checkpoint(run, checkpoint, context):
        calls = checkpoint['model_calls']
        if not number(calls) or calls < 0 or int(calls) != calls:
            raise ValueError('Invalid first-success TS checkpoint count')
        account = {**context, 'ts_checkpoint_calls': calls,
                   'per_reaction_first_success_calls': context['per_reaction_pre_ts_calls'] + calls,
                   'campaign_first_success_calls': context['campaign_pre_ts_calls'] + calls,
                   'cumulative_campaign_calls_at_attempt_first_success': context['campaign_pre_ts_calls'] + calls,
                   'campaign_first_success_calls_note': 'Legacy alias for cumulative_campaign_calls_at_attempt_first_success; '
                                                        'this is the current attempt event, not necessarily the earliest '
                                                        'campaign or reaction success across retries',
                   'ts_checkpoint': checkpoint}
        atomic_json(Path(run) / 'first_success_accounting.json', account)
        return account


class Campaign:
    def __init__(self, out, request, resume=False, engine=None):
        self.out, self.request = Path(out), request
        self.engine = engine or Pipeline()
        fingerprint = digest_json(request)
        if resume:
            self.state = read_json(self.out / 'state.json')
            if self.state['request_fingerprint'] != fingerprint:
                raise ValueError('Resume requires identical code, cohort, settings, params and seed')
            for name, value in (('request', request), ('cohort', request['cohort']),
                                ('settings', request['settings']), ('params', request['params'])):
                if read_json(self.out / (name + '.json')) != value:
                    raise ValueError(f'Frozen run input changed: {name}.json')
            self.verify_completed()
            if self.state['active']:
                self.finish_attempt('interrupted', error='Previous session ended before durable completion')
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            self.state = {'schema': 'chavis_general_v2_state_1', 'run_id': uuid.uuid4().hex,
                          'request_fingerprint': fingerprint, 'status': 'ready', 'active': None,
                          'preparation': {}, 'completed': {}, 'attempts': [], 'imported_charges': {}, 'created_epoch': time.time()}
            atomic_json(self.out / 'request.json', request)
            atomic_json(self.out / 'cohort.json', request['cohort'])
            atomic_json(self.out / 'settings.json', request['settings'])
            atomic_json(self.out / 'params.json', request['params'])
            self.save()

    def save(self):
        self.state['updated_epoch'] = time.time()
        atomic_json(self.out / 'state.json', self.state)
        self.write_costs()

    def charge_import(self, rid, sp, doc):
        """Charge the imported baseline reactant preparation exactly once per (seed, reaction, source checkpoint), outside
        attempt cost envelopes, so a retried preparation never charges it again; new CHAVIS calls of
        every attempt, failed ones included, stay in the attempt ledger."""
        external = self.request['external_reactants']
        source = external['pins'][rid]['checkpoint_sha256']
        key = f"seed{self.request['seed']}|{rid}|{source}"
        charges = self.state.setdefault('imported_charges', {})
        if key not in charges:
            charges[key] = {'key': key, 'reaction': rid, 'species': sp, 'seed': self.request['seed'],
                            'calls': int(doc['imported_reactant_calls']), 'seconds': doc.get('imported_reactant_seconds'),
                            'source_dir': doc['source_dir'], 'source_checkpoint_sha256': source,
                            'records': doc['n_records'], 'records_mapped': doc['n_mapped'],
                            'calls_definition': external['calls_definition'],
                            'charged_in_attempt': (self.state.get('active') or {}).get('path'), 'charged_epoch': time.time()}
            self.save()
            self.log('IMPORT_CHARGED', **{k: v for k, v in charges[key].items() if k != 'key'}, charge_key=key)
        return charges[key]

    def write_costs(self):
        attempts = self.state['attempts']
        abandoned = [a for a in attempts if a['status'] != 'complete']
        imported = sum(c['calls'] for c in self.state.get('imported_charges', {}).values())
        new = sum(a['known_calls'] for a in attempts)
        atomic_json(self.out / 'cost_ledger.json', {
            'known_total_calls': new + imported,
            'known_imported_calls': imported,
            'known_new_model_calls': new,
            'imported_charges': self.state.get('imported_charges', {}),
            'known_abandoned_calls': sum(a['known_calls'] for a in abandoned),
            'known_useful_calls': sum(a['known_calls'] for a in attempts if a['status'] == 'complete'),
            'known_stage_seconds': sum(a['known_stage_seconds'] for a in attempts),
            'known_surface_eos_calls': sum(a.get('known_surface_eos_calls', 0) for a in attempts),
            'known_surface_slab_calls': sum(a.get('known_surface_slab_calls', 0) for a in attempts),
            'known_abandoned_stage_seconds': sum(a['known_stage_seconds'] for a in abandoned),
            'unknown_cost_attempts': [a['path'] for a in attempts if not a['calls_complete']],
            'costs_complete': all(a['calls_complete'] and a['times_complete'] for a in attempts) and self.state['active'] is None,
            'attempts': attempts, 'active_attempt': self.state['active'],
            'scope': 'All stage attempts; completed preparation charged once; retries retain abandoned cost; '
                     'known_imported_calls = externally supplied reactant preparation (same-pool comparison), charged once per '
                     '(seed, reaction, source checkpoint) in imported_charges, included in known_total_calls; '
                     'known_new_model_calls = every attempt (completed and abandoned) of this run'})

    def log(self, event, **row):
        if 'key' in row:
            row['attempt_key'] = row.pop('key')
        with (self.out / 'ledger.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({'key': event, 'run_id': self.state['run_id'], 'epoch': time.time(), **row}, allow_nan=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    def verify_completed(self):
        groups = [self.state[group].values() for group in ('preparation', 'completed')]
        if self.state.get('surface_preparation'):
            groups.append([self.state['surface_preparation']])
        for entries in groups:
            for entry in entries:
                directory = (self.out / entry['path']).resolve()
                if not directory.is_relative_to(self.out.resolve()):
                    raise ValueError('Completed directory escapes run root')
                actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                          if p.is_file() and not any(x in p.relative_to(directory).parts
                              for x in ('logs', 'stage_requests', 'stage_costs', 'ts_attempts'))}
                if actual != set(entry['files']):
                    raise ValueError(f'Completed artifact file list changed: {directory}')
                for rel, digest in entry['files'].items():
                    path = (self.out / entry['path'] / rel).resolve()
                    if not path.is_relative_to(self.out.resolve()) or not path.is_file() or sha256(path) != digest:
                        raise ValueError(f'Completed artifact missing or changed: {path}')

    def start_attempt(self, kind, key):
        number_ = 1 + sum(a['kind'] == kind and a['key'] == key for a in self.state['attempts'])
        rel = f'{kind}/{key}/attempt_{number_:04d}'
        self.state['active'] = {'kind': kind, 'key': key, 'path': rel, 'started_epoch': time.time()}
        self.state['status'] = 'running'
        self.save()
        self.log('ATTEMPT_START', **self.state['active'])
        return self.out / rel

    def finish_attempt(self, status, payload=None, error=None):
        active = self.state['active']
        path = self.out / active['path']
        cost = cost_summary(path)
        record = {**active, **cost, 'status': status, 'ended_epoch': time.time(), 'error': error}
        if status == 'complete':
            files = {}
            # Completion fingerprints cover scientific output, not volatile stage logs.
            for p in sorted(path.rglob('*')):
                if p.is_file() and not any(x in p.relative_to(path).parts for x in ('logs', 'stage_requests', 'stage_costs', 'ts_attempts')):
                    files[p.relative_to(path).as_posix()] = sha256(p)
            entry = {**record, **(payload or {}), 'files': files}
            if active['kind'] == 'surface':
                self.state['surface_preparation'] = entry
            else:
                group = 'preparation' if active['kind'] == 'preparation' else 'completed'
                self.state[group][active['key']] = entry
        self.state['attempts'].append(record)
        self.state['active'] = None
        self.save()
        self.log('ATTEMPT_COMPLETE' if status == 'complete' else 'ATTEMPT_ABANDONED', **record)
        return record

    def execute(self):
        external = self.request.get('external_reactants')
        self.log('RUN_START', reactions=self.request['cohort']['reactions'],
                 external_reactants=None if not external else {k: external[k] for k in ('kind', 'seed_dir')})
        try:
            definition = self.request.get('surface', surface_definition())
            if definition['argument'] != DEFAULT_SURFACE and not self.state.get('surface_preparation'):
                if (self.out / 'STOP').exists():
                    self.state['status'] = 'stopped'
                    self.save()
                    self.log('RUN_STOPPED', next_stage='surface_preparation')
                    return 'stopped'
                path = self.start_attempt('surface', definition['key'])
                payload = self.engine.prepare_surface(self, path)
                self.finish_attempt('complete', payload)
            for row in self.request['cohort']['rows']:
                rid, sp = row['id'], row['parent_species']
                if rid in self.state['completed']:
                    continue
                if (self.out / 'STOP').exists():
                    self.state['status'] = 'stopped'
                    self.save()
                    self.log('RUN_STOPPED', next_reaction=rid)
                    return 'stopped'
                # External records are per reaction and seed, so preparation is never shared between reactions.
                key = f'{sp}__{rid}' if external else sp
                if key not in self.state['preparation']:
                    path = self.start_attempt('preparation', key)
                    payload = self.engine.prepare(self, sp, path, rid=rid) if external else self.engine.prepare(self, sp, path)
                    self.finish_attempt('complete', payload)
                prep = self.out / self.state['preparation'][key]['path']
                path = self.start_attempt('reactions', rid)
                payload = self.engine.reaction(self, rid, path, prep)
                self.finish_attempt('complete', payload)
                self.log('REACTION_' + rid, rid=rid, equation=row['execution_equation'], **payload)
            self.state['status'] = 'complete'
            self.save()
            self.log('RUN_COMPLETE', reactions=self.request['cohort']['reactions'])
            return 'complete'
        except BaseException as exc:
            if self.state['active']:
                self.finish_attempt('failed', error=repr(exc))
            self.state['status'] = 'failed'
            self.save()
            self.log('RUN_INCOMPLETE', error=repr(exc))
            raise
