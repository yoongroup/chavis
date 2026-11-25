"""Fresh Ni(111) campaign; only TS scheduling differs from first_success.py."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RIDS = 'D01 D02 D03 D04 D05 D06 D07 D08 D09 D10 D13 N01 N02 N03 N04'.split()
SURF, SURF_ARG = 'Ni_fcc111', 'Ni:fcc111'


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Campaign:
    # Kept as instance-overridable defaults for the legacy Ni(111) entrypoint.
    # The general runner supplies the requested surface before these checks run.
    surf, surf_arg = SURF, SURF_ARG

    def __init__(self, run_dir, settings):
        self.run = Path(run_dir).resolve()
        # mkdir is the atomic claim; even an existing empty directory is refused.
        self.run.mkdir(parents=True, exist_ok=False)
        (self.run / 'logs').mkdir()
        self.params = self.run / 'params.json'
        self.params.write_bytes((ROOT / 'reproduce/inputs/reference/params.json').read_bytes())
        self.settings = self.run / 'ultra_settings.json'
        self.settings.write_bytes(Path(settings).read_bytes())
        self.ledger = self.run / 'ledger.jsonl'
        self.run_id = time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        self.reactions = {r['id']: r for r in json.loads(
            (ROOT / 'campaigns/ni_drm/config/reactions.json').read_text())['reactions']}
        self.stage_records = []

    def write(self, **rec):
        with self.ledger.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(dict(run_id=self.run_id, **rec), allow_nan=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    def sp_dir(self, sp):
        return self.run / self.surf / sp

    @staticmethod
    def need(path, what):
        if not path.is_file():
            raise FileNotFoundError(f'{what}: {path}')
        return path

    def stage(self, key, tool, args, check, ultra=False):
        attempt = uuid.uuid4().hex[:12]
        cost = self.run / 'stage_costs' / f'{key}__{attempt}.json'
        if ultra:
            cmd = [sys.executable, '-B', '-m', 'ultra_fast_v1.cli', *map(str, args), '--cost-file', str(cost)]
        else:
            cmd = [sys.executable, '-B', str(HERE / 'stage.py'), str(cost), tool, *map(str, args)]
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'PYTHONDONTWRITEBYTECODE': '1'}
        t_wall, t_mono = time.time(), time.monotonic()
        with (self.run / 'logs' / f'{key}.log').open('ab') as log:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        elapsed = time.monotonic() - t_mono
        try:
            c = json.loads(cost.read_text())
            err = c.get('error') or (None if rc == 0 else f'rc {rc}')
            if not number(c.get('calls')) or c['calls'] < 0 or int(c['calls']) != c['calls']:
                err = err or 'missing or invalid model call count'
            if not number(c.get('seconds')) or c['seconds'] < 0:
                err = err or 'missing or invalid elapsed seconds'
        except Exception as exc:
            c, err = {}, f'cost file missing or unreadable: {exc!r}'
        if err is None:
            try:
                check(c)
            except Exception as exc:
                err = f'output check failed: {exc!r}'
        rec = dict(key=key, attempt=attempt, tool=tool, rc=rc, ok=err is None,
                   error=err, calls=c.get('calls'), seconds=c.get('seconds'),
                   subprocess_seconds=round(elapsed, 3), models_created=c.get('models_created'),
                   cost_file=str(cost.relative_to(self.run)), start_epoch=t_wall, end_epoch=time.time())
        self.write(**rec)
        self.stage_records.append(rec)
        if err:
            raise RuntimeError(f'stage {key} failed: {err}')

    def check_wf2(self, sp):
        if (self.sp_dir(sp) / 'error.txt').exists():
            raise RuntimeError(f'wf2 error recorded for {sp}')
        self.need(self.sp_dir(sp) / 'summary.json', 'wf2 summary')

    def check_select(self, sp):
        self.need(self.run / 'wf3_selection.json', 'selection')
        self.need(self.sp_dir(sp) / 'minima.json', 'minima')

    def check_wf3(self, sp):
        self.need(self.sp_dir(sp) / 'refined.jsonl', 'refined IS')
        summary = json.loads(self.need(self.run / 'wf3_is.json', 'wf3 summary').read_text())
        if self.surf not in summary['surfaces']:
            raise RuntimeError(f'wf3_is.json lacks requested surface {self.surf}')
        if {r['species'] for r in summary['surfaces'][self.surf].values()} - {sp}:
            raise RuntimeError(f'wf3_is.json holds other species than {sp}')
        part = self.run / 'wf3_parts' / f'{sp}.json'
        part.parent.mkdir(exist_ok=True)
        part.write_text(json.dumps(summary, indent=1), encoding='utf-8')

    def merge_wf3(self):
        parts = [json.loads(p.read_text()) for p in sorted((self.run / 'wf3_parts').glob('*.json'))]
        merged = dict(parts[-1])
        merged['surfaces'] = {self.surf: {}}
        for part in parts:
            if part['params'] != parts[0]['params']:
                raise RuntimeError('wf3 parameters changed between species')
            for rid, row in part['surfaces'].get(self.surf, {}).items():
                prev = merged['surfaces'][self.surf].get(rid)
                if prev is not None and prev != row:
                    raise RuntimeError(f'conflicting wf3 entries for {rid}')
                merged['surfaces'][self.surf][rid] = row
        (self.run / 'wf3_is.json').write_text(json.dumps(merged, indent=1), encoding='utf-8')

    def rep_pairs(self, rid):
        path = self.need(self.sp_dir(self.reactions[rid]['parent_species']) / f'pairs_{rid}.jsonl', 'pairs')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [row for row in rows if row['pair_representative']]

    def result_path(self, rid):
        return self.sp_dir(self.reactions[rid]['parent_species']) / f'ts_ultra_{rid}/reaction_result.json'

    def check_ultra(self, rid, cost):
        result = json.loads(self.need(self.result_path(rid), 'ultra reaction result').read_text())
        if result.get('status') != 'complete':
            raise RuntimeError(f'ultra reaction status {result.get("status")}')
        if result.get('model_calls') != cost['calls']:
            raise RuntimeError('reaction result and stage cost counts differ')
        if result.get('validated') and (not result.get('first_validated') or not number(result.get('best_barrier_eV'))):
            raise RuntimeError('validated reaction lacks a pair or finite global-reference barrier')
        scope = json.loads(self.settings.read_text(encoding='utf-8')).get('search_scope', 'first_success')
        if scope == 'all_pairs':
            pairs = self.rep_pairs(rid)
            expected = {f"pair_{pair['pair_group']:03d}" for pair in pairs}
            statuses = result.get('pair_statuses')
            if (result.get('search_scope') != scope or result.get('stop_reason') != 'all_pairs_complete'
                    or result.get('n_pairs') != len(expected) or result.get('pairs_complete') != len(expected)
                    or result.get('pairs_run') != len(expected) or not isinstance(statuses, list)
                    or len(statuses) != len(expected)
                    or {row.get('pair') for row in statuses} != expected):
                raise RuntimeError('all_pairs result does not cover every representative pair')
            if (result.get('pending_candidates') != 0 or result.get('interrupted_candidates') != 0
                    or any(row.get('complete') is not True or row.get('prepared') is not True
                           or row.get('pending_candidates') != 0 or row.get('interrupted_candidate') is not None
                           or row.get('completion_reason') not in ('validated_share', 'candidates_exhausted')
                           for row in statuses)):
                raise RuntimeError('all_pairs result retains a nonterminal pair or pending candidate')

    def execute(self):
        self.write(key='RUN_START', mode='ultra-fast', start_epoch=time.time(), rids=RIDS)
        species_done = set()
        for rid in RIDS:
            first_stage = len(self.stage_records)
            sp = self.reactions[rid]['parent_species']
            if sp not in species_done:
                self.stage(f'wf2_{sp}', 'run_wf2.py', ['--surface', self.surf_arg, '--species', sp,
                           '--out', self.run, '--mode', 'fast', '--fix-layers', 'all',
                           '--store', ROOT / 'slab_store'], lambda c, sp=sp: self.check_wf2(sp))
                self.stage(f'select_after_{sp}', 'select_wf3.py', [self.run, '--params', self.params, '--rebuild'],
                           lambda c, sp=sp: self.check_select(sp))
                self.stage(f'wf3_{sp}', 'run_wf3.py', [self.run, '--params', self.params, '--surface', self.surf, '--species', sp],
                           lambda c, sp=sp: self.check_wf3(sp))
                species_done.add(sp)
            self.merge_wf3()
            if rid not in json.loads((self.run / 'wf3_is.json').read_text())['surfaces'][self.surf]:
                raise RuntimeError(f'{rid} missing from merged wf3_is.json')
            self.stage(f'wf4_{rid}', 'run_wf4.py', [self.run, '--surface', self.surf, '--reaction', rid, '--params', self.params],
                       lambda c, rid=rid, sp=sp: self.need(self.sp_dir(sp) / f'fs_{rid}.jsonl', 'FS file'))
            self.stage(f'pairs_{rid}', 'run_wf5_pairs.py', [self.run, '--surface', self.surf, '--reaction', rid, '--params', self.params],
                       lambda c, rid=rid: self.rep_pairs(rid))
            self.stage(f'ts_{rid}_ultra', 'ultra_fast_v1.cli', [self.run, '--params', self.params,
                       '--settings', self.settings, '--surface', self.surf, '--reaction', rid],
                       lambda c, rid=rid: self.check_ultra(rid, c), ultra=True)
            result = json.loads(self.result_path(rid).read_text())
            self.write(key=f'REACTION_{rid}', rid=rid, equation=self.reactions[rid]['equation'], species=sp,
                       n_pairs=result.get('n_pairs'), pairs_run=result.get('pairs_run'),
                       first_validated=result.get('first_validated'), validated=result.get('validated'),
                       best_barrier_eV=result.get('best_barrier_eV'), ts_calls=result['model_calls'],
                       calls=sum(r['calls'] for r in self.stage_records[first_stage:]),
                       calls_scope='stages since previous reaction; shared preparation charged on first use',
                       result_file=str(self.result_path(rid).relative_to(self.run)), end_epoch=time.time())
        self.write(key='RUN_COMPLETE', end_epoch=time.time(), calls=sum(r['calls'] for r in self.stage_records))


def main(argv=None):
    sys.path.insert(0, str(ROOT))
    from generic_v2.common import RUN_ROOT, output_path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', help=f'Fresh physical output directory below {RUN_ROOT}')
    ap.add_argument('--settings', default=str(ROOT / 'reproduce/inputs/ultra_fast/settings.json'))
    args = ap.parse_args(argv)
    if os.name == 'posix':
        output_path(args.run_dir, allowed=RUN_ROOT)
    campaign = Campaign(args.run_dir, args.settings)
    sys.path.insert(0, str(ROOT / 'src'))
    from chavis.runtime.run_snapshot import start_run, end_run
    snap_started = False
    try:
        start_run(campaign.run, ROOT, [sys.executable, *sys.argv],
                  config={'params': json.loads(campaign.params.read_text()), 'ultra': json.loads(campaign.settings.read_text())},
                  input_paths=[campaign.params, campaign.settings, ROOT / 'slab_store'],
                  workers=1, code_entries=('src', 'tools', 'config', 'campaigns/ni_drm/config',
                      'support/runtime_support/automation_search', 'reproduce/drivers',
                      'reproduce/inputs/reference', 'reproduce/inputs/ultra_fast', 'ultra_fast_v1',
                      'launch_ultra_fast.sh'))
        snap_started = True
        campaign.execute()
        end_run(campaign.run, 'complete')
    except BaseException as exc:
        campaign.write(key='RUN_INCOMPLETE', reason=repr(exc), end_epoch=time.time())
        if snap_started:
            end_run(campaign.run, 'error', repr(exc))
        raise
    print('ultra-fast run done', flush=True)


if __name__ == '__main__':
    main()
