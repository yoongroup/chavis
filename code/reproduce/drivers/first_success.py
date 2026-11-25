import json, os, subprocess, sys, time, uuid
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / '../'
PY = os.environ.get('CHAVIS_PYTHON', sys.executable)
SURF, SURF_ARG = ('Ni_fcc111', 'Ni:fcc111')
RIDS = 'D01 D02 D03 D04 D05 D06 D07 D08 D09 D10 D13 N01 N02 N03 N04'.split()
TASK2 = Path('reproduce/inputs/reference')
run = Path(sys.argv[1]).resolve()
if run.exists() and any(run.iterdir()):
    raise SystemExit(f'{run} is not empty: compare runs are one-shot, use a new directory (S03/S04)')
run.mkdir(parents=True, exist_ok=True)
(run / 'logs').mkdir()
params = run / 'params.json'
params.write_text((TASK2 / 'params.json').read_text())
assert json.loads(params.read_text()) == json.loads((TASK2 / 'params.json').read_text()), 'params differ from A0'
ledger = run / 'ledger.jsonl'
reactions = {r['id']: r for r in json.loads((ROOT / 'campaigns/ni_drm/config/reactions.json').read_text())['reactions']}
RUN_ID = time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]


def write(rec):
    with ledger.open('a') as fh:
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def stage(key, tool, args, check):
    attempt = uuid.uuid4().hex[:12]
    cost = run / 'stage_costs' / f'{key}__{attempt}.json'
    before = set(os.listdir(run / 'ts_attempts')) if (run / 'ts_attempts').is_dir() else set()
    t_wall, t_mono = (time.time(), time.monotonic())
    rc = subprocess.run(
        [PY, '-B', str(HERE / 'stage.py'), str(cost), tool, *map(str, args)],
        env={
            **os.environ,
            'HF_HUB_OFFLINE': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'PYTHONDONTWRITEBYTECODE': '1'
        },
        stdout=open(run / 'logs' / f'{key}.log', 'ab'),
        stderr=subprocess.STDOUT
    ).returncode
    elapsed = time.monotonic() - t_mono
    try:
        c = json.loads(cost.read_text())
        err = c.get('error') or (None if rc == 0 else f'rc {rc}')
    except Exception as e:
        c, err = ({}, f'cost file missing or unreadable: {e!r}')
    if err is None:
        try:
            check(before, c)
        except Exception as e:
            err = f'output check failed: {e!r}'
    write(dict(
        run_id=RUN_ID,
        key=key,
        attempt=attempt,
        tool=tool,
        rc=rc,
        ok=err is None,
        error=err,
        calls=c.get('calls'),
        seconds=c.get('seconds'),
        subprocess_seconds=round(elapsed, 3),
        models_created=c.get('models_created'),
        cost_file=str(cost.relative_to(run)),
        start_epoch=t_wall,
        end_epoch=time.time()
    ))
    if err:
        write(dict(
            run_id=RUN_ID,
            key='RUN_INCOMPLETE',
            reason=f'{key}: {err}',
            end_epoch=time.time()
        ))
        raise SystemExit(f'stage {key} failed: {err}')


def sp_dir(sp):
    return run / SURF / sp


def need(path, what):
    if not path.is_file():
        raise FileNotFoundError(f'{what}: {path}')
    return path


def check_wf2(sp):

    def f(*_):
        if (sp_dir(sp) / 'error.txt').exists():
            raise RuntimeError(f'wf2 error recorded for {sp}')
        need(sp_dir(sp) / 'summary.json', 'wf2 summary')
    return f


def check_select(sp):

    def f(*_):
        need(run / 'wf3_selection.json', 'selection')
        need(sp_dir(sp) / 'minima.json', 'minima')
    return f


def species_rids(sp):
    return [r for r in RIDS if reactions[r]['parent_species'] == sp]


def check_wf3(sp):

    def f(*_):
        need(sp_dir(sp) / 'refined.jsonl', 'refined IS')
        summary = json.loads(need(run / 'wf3_is.json', 'wf3 summary').read_text())
        rows = summary['surfaces'].get(SURF, {})
        if {row['species'] for row in rows.values()} - {sp}:
            raise RuntimeError(f'wf3_is.json holds other species than {sp}')
        part = run / 'wf3_parts' / f'{sp}.json'
        part.parent.mkdir(exist_ok=True)
        part.write_text(json.dumps(summary, indent=1))
    return f


def merge_wf3():
    parts = [json.loads(p.read_text()) for p in sorted((run / 'wf3_parts').glob('*.json'))]
    merged = dict(parts[-1])
    merged['surfaces'] = {SURF: {}}
    for part in parts:
        assert part['params'] == parts[0]['params']
        for rid, row in part['surfaces'].get(SURF, {}).items():
            assert rid not in merged['surfaces'][SURF] or merged['surfaces'][SURF][rid] == row, rid
            merged['surfaces'][SURF][rid] = row
    (run / 'wf3_is.json').write_text(json.dumps(merged, indent=1))


def check_file(path, what):
    return lambda *_: need(path, what)


def rep_pairs(rid):
    p = need(
        sp_dir(reactions[rid]['parent_species']) / f'pairs_{rid}.jsonl',
        'pairs file'
    )
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return sorted(
        (r for r in rows if r['pair_representative']),
        key=lambda r: (r['is_energy_eV'], r['reaction_energy_eV'])
    )


def ts_dir(rid):
    return sp_dir(reactions[rid]['parent_species']) / f'ts_fast_{rid}'


def check_ts(rid, k, verify=False):

    def f(before, c):
        new = sorted(set(os.listdir(run / 'ts_attempts')) - before)
        if len(new) != 1:
            raise RuntimeError(f'expected one new TS attempt, found {new}')
        rec = json.loads(need(
            run / 'ts_attempts' / new[0] / '_snapshot' / 'run.json',
            'attempt record'
        ).read_text())
        if rec.get('status') != 'complete':
            raise RuntimeError(f"TS attempt {new[0]} status {rec.get('status')}")
        argv = list(map(str, rec['argv']))
        opt = lambda name: argv[argv.index(name) + 1] if name in argv else None
        if opt('--reaction') != rid or opt('--max-pairs') != str(k):
            raise RuntimeError(f'TS attempt {new[0]} argv does not match {rid} k={k}: {argv}')
        if verify and (c.get('calls') != 0 or c.get('models_created') != 0):
            raise RuntimeError(f"verify pass computed: calls {c.get('calls')}, models {c.get('models_created')}")
        errs = sorted(ts_dir(rid).glob('pair_*/error.json'))
        if errs:
            raise RuntimeError(f'TS pair error recorded: {errs[0]}')
        for p in rep_pairs(rid)[:k]:
            res = json.loads(need(
                ts_dir(rid) / f"pair_{p['pair_group']:03d}" / 'result.json',
                'TS result'
            ).read_text())
            if not res.get('request_fingerprint') or not res.get('attempt_id'):
                raise RuntimeError(f"pair {p['pair_group']}: result without fingerprint/attempt")
        if not verify and k and (json.loads((ts_dir(rid) / f"pair_{rep_pairs(rid)[k - 1]['pair_group']:03d}" / 'result.json').read_text())['attempt_id'] != new[0]):
            raise RuntimeError(f'pair {k} result not written by attempt {new[0]}')
    return f


def validated_pair(rid, k):
    for p in rep_pairs(rid)[:k]:
        c = ts_dir(rid) / f"pair_{p['pair_group']:03d}" / 'candidates.jsonl'
        if c.exists() and any(((json.loads(l).get('judged') or {}).get('validated') for l in c.read_text().splitlines() if l.strip())):
            return f"pair_{p['pair_group']:03d}"
    return None


def ts_args(rid, k):
    return [
        run,
        '--params',
        params,
        '--surface',
        SURF,
        '--reaction',
        rid,
        '--mode',
        'fast',
        '--dimer-max-steps',
        1500,
        '--max-pairs',
        k
    ]


write(dict(run_id=RUN_ID, key='RUN_START', start_epoch=time.time(), rids=RIDS))
species_done = set()
for rid in RIDS:
    sp = reactions[rid]['parent_species']
    if sp not in species_done:
        stage(
            f'wf2_{sp}',
            'run_wf2.py',
            [
                '--surface',
                SURF_ARG,
                '--species',
                sp,
                '--out',
                run,
                '--mode',
                'fast',
                '--fix-layers',
                'all',
                '--store',
                ROOT / 'slab_store'
            ],
            check_wf2(sp)
        )
        stage(
            f'select_after_{sp}',
            'select_wf3.py',
            [run, '--params', params, '--rebuild'],
            check_select(sp)
        )
        stage(
            f'wf3_{sp}',
            'run_wf3.py',
            [run, '--params', params, '--surface', SURF, '--species', sp],
            check_wf3(sp)
        )
        species_done.add(sp)
    merge_wf3()
    if rid not in json.loads((run / 'wf3_is.json').read_text())['surfaces'][SURF]:
        write(dict(
            run_id=RUN_ID,
            key='RUN_INCOMPLETE',
            reason=f'{rid} missing from merged wf3_is.json',
            end_epoch=time.time()
        ))
        raise SystemExit(f'{rid} missing from merged wf3_is.json')
    stage(
        f'wf4_{rid}',
        'run_wf4.py',
        [run, '--surface', SURF, '--reaction', rid, '--params', params],
        check_file(sp_dir(sp) / f'fs_{rid}.jsonl', 'FS file')
    )
    stage(
        f'pairs_{rid}',
        'run_wf5_pairs.py',
        [run, '--surface', SURF, '--reaction', rid, '--params', params],
        lambda *_,
        rid=rid: rep_pairs(rid)
    )
    total, hit, k = (len(rep_pairs(rid)), None, 0)
    while hit is None and k < total:
        k += 1
        stage(
            f'ts_{rid}_upto{k:03d}',
            'run_wf5_ts.py',
            ts_args(rid, k),
            check_ts(rid, k)
        )
        hit = validated_pair(rid, k)
    if k:
        stage(
            f'ts_{rid}_verify{k:03d}',
            'run_wf5_ts.py',
            ts_args(rid, k),
            check_ts(rid, k, verify=True)
        )
        if validated_pair(rid, k) != hit:
            write(dict(
                run_id=RUN_ID,
                key='RUN_INCOMPLETE',
                reason=f'{rid}: validated pair changed after verify',
                end_epoch=time.time()
            ))
            raise SystemExit(f'{rid}: validated pair changed after verify')
    write(dict(
        run_id=RUN_ID,
        key=f'REACTION_{rid}',
        rid=rid,
        species=sp,
        n_pairs=total,
        pairs_run=k,
        first_validated=hit,
        end_epoch=time.time()
    ))
write(dict(run_id=RUN_ID, key='RUN_COMPLETE', end_epoch=time.time()))
print('compare run done')
