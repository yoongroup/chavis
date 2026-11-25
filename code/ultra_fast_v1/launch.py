"""Offline campaign supervisor. Importing this module never starts a calculation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time



ROOT = Path(__file__).resolve().parents[1]
from generic_v2.common import HF_CACHE, PYTHON as VENV_PYTHON, REVISION, RUN_ROOT, output_path
RIDS = 'D01 D02 D03 D04 D05 D06 D07 D08 D09 D10 D13 N01 N02 N03 N04'.split()
PAUSE_C, RESUME_C = 85, 75
SIGSTOP = getattr(signal, 'SIGSTOP', 19)
SIGCONT = getattr(signal, 'SIGCONT', 18)
SIGKILL = getattr(signal, 'SIGKILL', 9)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def verify_environment(root=ROOT, python=VENV_PYTHON, cache=HF_CACHE, surface='Ni:fcc111'):
    if not Path(python).is_file():
        raise RuntimeError(f'Python interpreter missing: {python}')
    snapshot = Path(cache) / 'models--facebook--UMA/snapshots' / REVISION
    for rel in ('checkpoints/uma-s-1p2p1.pt', 'references/iso_atom_elem_refs.yaml', 'references/form_elem_refs.yaml'):
        if not (snapshot / rel).is_file():
            raise RuntimeError(f'pinned offline model asset missing: {snapshot / rel}')
    # Nondefault surfaces may require their first EOS/slab preparation. Present
    # caches must match the exact surface, model, specification and geometry.
    if str(Path(root) / 'src') not in sys.path:
        sys.path.insert(0, str(Path(root) / 'src'))
    from generic_v2.surfaces import DEFAULT_SURFACE, inspect_cache
    status = inspect_cache(Path(root) / 'slab_store', surface)
    if surface == DEFAULT_SURFACE and not status['lattice_cached']:
        raise RuntimeError('Ni UMA lattice cache missing')
    if surface == DEFAULT_SURFACE and not status['slab_cached']:
        raise RuntimeError('Ni(111) slab record missing')


def ledger_complete(path, require_validated=False):
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError):
        return False
    if any(r.get('key') == 'RUN_INCOMPLETE' or r.get('ok') is False for r in rows):
        return False
    reactions = [r for r in rows if r.get('key', '').startswith('REACTION_')]
    if len(reactions) != len(RIDS) or {r.get('rid') for r in reactions} != set(RIDS):
        return False
    if require_validated and not all(r.get('first_validated') for r in reactions):
        return False
    return bool(rows and rows[-1].get('key') == 'RUN_COMPLETE')


def predecessor_ready(after, log=None):
    after = Path(after)
    if not after.is_dir():
        return False
    # Only the last campaign status marker is authoritative; a subsequent START
    # invalidates a stale DONE line in a reused log.
    statuses = []
    if log is not None:
        try:
            statuses = [line.split()[0] for line in Path(log).read_text(errors='replace').splitlines()
                        if line.startswith(('FOR_SUBMIT_STARTED ', 'FOR_SUBMIT_DONE ', 'FOR_SUBMIT_FAILED '))]
        except OSError:
            pass
    # Bind the completion marker to an existing three-run campaign. A typo or
    # unrelated empty --after directory must not consume a stale global marker.
    has_three_runs = all((after / f'fs_r{i}').is_dir() for i in range(1, 4))
    if has_three_runs and statuses and statuses[-1] == 'FOR_SUBMIT_DONE':
        return True
    return all(ledger_complete(after / f'fs_r{i}/ledger.jsonl') for i in range(1, 4))


class ProbeUnavailable(RuntimeError):
    pass


def probe_temperature(run=subprocess.run, sleep=time.sleep, alive=None):
    """30-second timeout, an initial attempt plus up to three retries."""
    errors = []
    for attempt in range(4):
        if alive is not None and not alive():
            return None  # Child completion wins over a late watcher failure.
        try:
            result = run(['nvidia-smi', '-i', '0', '--query-gpu=temperature.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True,
                         text=True, timeout=30, check=True)
            value = float(result.stdout.strip().splitlines()[0])
            if not math.isfinite(value) or not 0 <= value <= 150:
                raise ValueError(f'invalid temperature: {value}')
            return value
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            errors.append(repr(exc))
        if attempt != 3:
            sleep(2)
    raise ProbeUnavailable('; '.join(errors))


def thermal_state(paused, temperature):
    if temperature is None or temperature >= PAUSE_C:
        return True
    if paused and temperature > RESUME_C:
        return True
    return False


def signal_owned(children, sig, killpg=os.killpg if hasattr(os, 'killpg') else None):
    """Every child has start_new_session=True; never discover unrelated PIDs."""
    for process in children.values():
        if process.poll() is None:
            try:
                killpg(process.pid, sig)
            except ProcessLookupError:
                pass


def command_for(run, root=ROOT, python=VENV_PYTHON, method='ultra-fast'):
    driver = {'ultra-fast': 'ultra_fast.py', 'first-success': 'first_success.py'}[method]
    return [str(python), '-B', str(Path(root) / 'reproduce/drivers' / driver), str(run)]


def supervise(children, log, probe=probe_temperature, sleep=time.sleep,
              monotonic=time.monotonic, send=signal_owned):
    """Return actual child exit codes; temperature errors only pause work."""
    paused, next_probe = False, 0.0
    alive = lambda: any(p.poll() is None for p in children.values())
    while alive():
        now = monotonic()
        if now >= next_probe:
            try:
                temperature = probe(alive=alive)
                probe_error = None
            except ProbeUnavailable as exc:
                temperature, probe_error = None, str(exc)
            # Probe may take several seconds while all jobs finish. Reap first.
            if not alive():
                break
            next_paused = thermal_state(paused, temperature)
            if next_paused != paused:
                send(children, SIGSTOP if next_paused else SIGCONT)
            paused = next_paused
            log({'event': 'thermal', 'temperature_C': temperature, 'paused': paused, 'probe_error': probe_error})
            next_probe = monotonic() + 60
        sleep(2)
    return {name: process.wait() for name, process in children.items()}


def launch(args):
    if args.runs not in (1, 3):
        raise ValueError('ultra-fast v2 supports --runs 1 or 3')
    if os.name != 'posix':
        raise RuntimeError('launch in WSL/Linux; Windows is supported only for CPU tests')
    out = output_path(args.out, allowed=RUN_ROOT)
    after = Path(args.after).resolve() if args.after else None
    after_log = Path(args.after_log).resolve() if args.after_log else None
    if after is not None and (out == after or out.is_relative_to(after) or after.is_relative_to(out)):
        raise ValueError('new output and predecessor output must be separate trees')
    verify_environment()
    out.mkdir(parents=True, exist_ok=False)  # Reserve output atomically before waiting.
    logfile = out / 'launcher.jsonl'

    def log(record):
        with logfile.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({'epoch': time.time(), **record}, allow_nan=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        print(json.dumps(record), flush=True)

    log({'event': 'waiting' if after else 'ready', 'after': str(after) if after else None, 'completion_log': str(after_log) if after_log else None,
         'runs': args.runs, 'method': args.method, 'pause_C': PAUSE_C, 'resume_C': RESUME_C,
         'python': str(VENV_PYTHON), 'HF_HUB_CACHE': str(HF_CACHE), 'model_revision': REVISION,
         'argv': sys.argv})
    while after is not None and not predecessor_ready(after, after_log):
        time.sleep(30)
    # Conditions and immutable files may have changed during the wait.
    verify_environment()
    startup_paused = False
    while True:
        try:
            temperature = probe_temperature()
            error = None
        except ProbeUnavailable as exc:
            temperature, error = None, str(exc)
        startup_paused = thermal_state(startup_paused, temperature)
        log({'event': 'startup_thermal', 'temperature_C': temperature,
             'paused': startup_paused, 'probe_error': error})
        if not startup_paused:
            break
        time.sleep(60)
    env = {**os.environ, 'PATH': str(VENV_PYTHON.parent) + os.pathsep + os.environ.get('PATH', ''),
           'HF_HUB_CACHE': str(HF_CACHE), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1'}
    children, streams = {}, []
    try:
        for i in range(1, args.runs + 1):
            name = f"{'uf' if args.method == 'ultra-fast' else 'fs'}_r{i}"
            stream = (out / f'{name}.out').open('xb')
            streams.append(stream)
            cmd = command_for(out / name, method=args.method)
            children[name] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                                              stderr=subprocess.STDOUT, start_new_session=True)
            log({'event': 'started', 'name': name, 'pid': children[name].pid, 'argv': cmd})
        codes = supervise(children, log)
        checks = {name: ledger_complete(out / name / 'ledger.jsonl') for name in children}
        result = {'returncodes': codes, 'complete_ledgers': checks,
                  'ok': all(code == 0 for code in codes.values()) and all(checks.values())}
        (out / 'campaign_end.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        log({'event': 'complete' if result['ok'] else 'failed', **result})
        return 0 if result['ok'] else 1
    except BaseException as exc:
        log({'event': 'supervisor_error', 'error': repr(exc)})
        # Resume before terminating: SIGTERM is not handled while stopped.
        signal_owned(children, SIGCONT)
        signal_owned(children, signal.SIGTERM)
        for process in children.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_owned({'process': process}, SIGKILL)
                process.wait()
        raise
    finally:
        for stream in streams:
            stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', help='Optional completed three-run campaign prerequisite')
    parser.add_argument('--after-log', type=Path, default=None,
                        help='Optional predecessor completion log; without it, check completed ledgers only')
    parser.add_argument('--out', required=True, help=f'Fresh physical output directory below {RUN_ROOT}')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--method', choices=('ultra-fast', 'first-success'), default='ultra-fast')
    args = parser.parse_args(argv)
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        # Raise into launch's cleanup rather than orphaning paused GPU jobs.
        raise KeyboardInterrupt(f'supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    try:
        return launch(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
