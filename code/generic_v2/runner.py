"""One sequential v2 campaign, with thermal supervision and reaction-boundary resume."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from .common import (CODE_ENTRIES, HF_CACHE, PYTHON, REVISION, RUN_ROOT, atomic_json,
                     cohort, environment, output_path, read_json, request)
from .campaign import Campaign
from .process import parent_death_options
from .surfaces import DEFAULT_SURFACE, SURFACES
from ultra_fast_v1.launch import (ProbeUnavailable, SIGCONT, SIGKILL, probe_temperature,
                                  signal_owned, supervise, thermal_state, verify_environment)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', help=f'Physical output directory below {RUN_ROOT}')
    p.add_argument('--cohort', default='paper15', help='paper15 (default) or JSON path containing reactions list')
    p.add_argument('--settings', default=str(ROOT / 'reproduce/inputs/ultra_fast/settings.json'))
    p.add_argument('--surface', choices=SURFACES, default=DEFAULT_SURFACE)
    p.add_argument('--search-scope', choices=('first_success', 'all_pairs'), default=None,
                   help='Optional settings override: first success globally, or first success for every representative pair')
    p.add_argument('--seed', type=int, default=1, choices=(1, 2, 3), help='Independent execution replicate label (1-3); unchanged frozen embedding seed61453')
    p.add_argument('--reactant-records', default=None,
                   help='Same-pool comparison: baseline run root (holds seed1..3/reactions/<rid>/checkpoint.json); '
                        'its relaxed reactant records of seed = --seed replace the wf2 adsorption search per reaction')
    p.add_argument('--resume', action='store_true', help='Resume same output, cohort, settings and code at reaction boundaries')
    p.add_argument('--preflight', action='store_true', help='CPU-only build/config/cohort validation; no GPU probe or model load')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return p


def diagnostic(req):
    rows = req['cohort']['rows']
    return {'protocol': req['schema'], 'cohort': req['cohort']['name'], 'reactions': req['cohort']['reactions'],
            'reaction_count': len(rows), 'parent_species': sorted({r['parent_species'] for r in rows}),
            'parent_species_count': len({r['parent_species'] for r in rows}),
            'execution_equations': {r['id']: r['execution_equation'] for r in rows},
            'wf2_azimuth_argument': ['--adsorption-azimuth-step-deg', str(req['settings']['adsorption_azimuth_step_deg'])],
            'seed': req['seed'], 'seed_semantics': req['seed_semantics'],
            'model_revision': REVISION, 'model_processes': 1,
            'surface': req.get('surface'), 'surface_cache': req.get('surface_cache'),
            'surface_preparation_accounting': 'separate EOS and clean slab stage; run-local cache for additional surfaces',
            'search_scope': req['settings'].get('search_scope', 'first_success'),
            'preparation_reaction_scope': ('one reaction per preparation (external reactant records)' if req.get('external_reactants')
                                           else 'requested cohort IDs for each parent species'),
            'external_reactants': ({k: req['external_reactants'][k] for k in ('kind', 'seed_dir', 'available', 'calls_definition')}
                                   if req.get('external_reactants') else None),
            'model_loaded': False}


def cleanup_groups(children, killpg=None, sleep=time.sleep):
    """The owned group can survive its leader; never gate cleanup on poll()."""
    killpg = killpg or os.killpg
    remaining = set()
    for proc in children.values():
        try:
            killpg(proc.pid, SIGCONT)
            killpg(proc.pid, signal.SIGTERM)
            remaining.add(proc.pid)
        except ProcessLookupError:
            pass
    # No new discovery/PID scan: these PGIDs came only from start_new_session.
    for _ in range(10):
        if not remaining:
            break
        sleep(1)
        for pgid in tuple(remaining):
            try:
                killpg(pgid, 0)
            except ProcessLookupError:
                remaining.remove(pgid)
    for pgid in remaining:
        try:
            killpg(pgid, SIGKILL)
        except ProcessLookupError:
            pass
    for proc in children.values():
        proc.wait()


def worker(args, req):
    # The inherited descriptor keeps the GPU lock alive if the supervisor dies.
    descriptor = int(os.environ['CHAVIS_GENERAL_GPU_LOCK_FD'])
    os.fstat(descriptor)
    out = output_path(args.out)
    campaign = Campaign(out, req, resume=True)
    session = out / 'sessions' / (time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:12])
    # CUDA, plotting and temporary caches all belong to the run session.
    env = environment()
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR',
                'TRITON_CACHE_DIR', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH'):
        path = session / 'runtime' / key.lower()
        path.mkdir(parents=True)
        env[key] = str(path)
    env['CUDA_CACHE_DISABLE'] = '1'
    os.environ.update(env)
    os.chdir(ROOT)
    from chavis.runtime.run_snapshot import start_run, end_run
    checkpoint = HF_CACHE / 'models--facebook--UMA/snapshots' / REVISION / 'checkpoints/uma-s-1p2p1.pt'
    start_run(session, ROOT, [sys.executable, *sys.argv], config=req,
              input_paths=[out / 'request.json', out / 'cohort.json', out / 'settings.json', out / 'params.json',
                           ROOT / 'campaigns/ni_drm/config', ROOT / 'slab_store', checkpoint],
              model={'name': 'uma-s-1p2p1', 'revision': REVISION, 'task': 'oc20', 'checkpoint': str(checkpoint)},
              workers={'gpu_processes_at_once': 1, 'reaction_parallelism': 1}, code_entries=CODE_ENTRIES,
              note='Reaction-boundary resume; completed reactions and shared species preparation are immutable.')
    try:
        status = campaign.execute()
        end_run(session, status)
        return 0
    except BaseException as exc:
        end_run(session, 'failed', repr(exc))
        raise


def launch(args, req):
    if sys.platform != 'linux':
        raise RuntimeError('GPU execution requires Linux/WSL; use --preflight for CPU verification')
    import fcntl
    out = output_path(args.out)
    verify_environment(ROOT, PYTHON, HF_CACHE, surface=req['surface']['argument'])
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    # An OS lock is automatically released on crash. Never infer ownership from a stale PID.
    with (RUN_ROOT / '.chavis_general_v2_gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another general-v2 launcher holds the single GPU slot') from exc
        campaign = Campaign(out, req, resume=args.resume)
        def log(row):
            with (out / 'launcher.jsonl').open('a', encoding='utf-8') as fh:
                fh.write(json.dumps({'epoch': time.time(), **row}, allow_nan=False) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
            print(json.dumps(row), flush=True)
        log({'event': 'ready', 'resume': args.resume, 'pause_C': 85, 'resume_C': 75,
             'python': str(PYTHON), 'HF_HUB_CACHE': str(HF_CACHE), **diagnostic(req)})
        if campaign.state['status'] == 'complete':
            log({'event': 'already_complete'})
            return 0
        paused = False
        while True:
            if (out / 'STOP').exists():
                campaign.state['status'] = 'stopped'
                campaign.save()
                log({'event': 'stopped_before_worker'})
                return 0
            try:
                temp, error = probe_temperature(), None
            except ProbeUnavailable as exc:
                temp, error = None, str(exc)
            paused = thermal_state(paused, temp)
            log({'event': 'startup_thermal', 'temperature_C': temp, 'paused': paused, 'error': error})
            if not paused:
                break
            for _ in range(30):
                if (out / 'STOP').exists():
                    break
                time.sleep(2)
        cmd = [str(PYTHON), '-B', '-m', 'generic_v2.runner', '--worker', '--out', str(out),
               '--cohort', args.cohort, '--settings', str(Path(args.settings).resolve()), '--seed', str(args.seed),
               '--surface', args.surface]
        if args.search_scope is not None:
            cmd += ['--search-scope', args.search_scope]
        if args.reactant_records:
            cmd += ['--reactant-records', str(Path(args.reactant_records).resolve())]
        children = {}
        with (out / 'worker.out').open('ab') as stream:
            try:
                env = {**environment(), 'CHAVIS_GENERAL_GPU_LOCK_FD': str(lock.fileno())}
                children['campaign'] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                                                        stderr=subprocess.STDOUT, start_new_session=True,
                                                        pass_fds=(lock.fileno(),), **parent_death_options())
                log({'event': 'worker_started', 'pid': children['campaign'].pid, 'argv': cmd})
                codes = supervise(children, log)
                cleanup_groups(children)
                state = read_json(out / 'state.json')
                good = codes['campaign'] == 0 and state['status'] in ('complete', 'stopped')
                result = {'returncodes': codes, 'status': state['status'], 'ok': good}
                atomic_json(out / 'campaign_end.json', result)
                log({'event': 'worker_finished', **result})
                return 0 if good else 1
            except BaseException as exc:
                log({'event': 'supervisor_error', 'error': repr(exc)})
                cleanup_groups(children)
                raise


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cohort not in ('paper15',):
        args.cohort = str(Path(args.cohort).resolve())
    req = request(cohort(args.cohort), args.settings, args.seed, reactant_records=args.reactant_records,
                  surface=args.surface, search_scope=args.search_scope)
    if args.preflight:
        print(json.dumps(diagnostic(req), indent=2))
        return 0
    if not args.out:
        raise ValueError('--out is required unless --preflight is used')
    if args.reactant_records and not req['external_reactants']['available']:
        raise FileNotFoundError(f"baseline records for seed {args.seed} not found under {args.reactant_records}")
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'general-v2 received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    try:
        return worker(args, req) if args.worker else launch(args, req)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
