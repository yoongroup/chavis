"""Same-pool comparison: import one reaction's baseline reactant records as the wf2 result of its parent species.
Runs under reproduce/drivers/stage.py so that the few model calls (gas-phase molecule) are counted like any stage.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '2')
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'support' / 'runtime_support')]
from chavis.surface import slab_store
from chavis.inputs.external_reactants import import_records, resolve_records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--surface', required=True, help='Element:facet, e.g. Ni:fcc111')
    ap.add_argument('--species', required=True)
    ap.add_argument('--reaction', required=True)
    ap.add_argument('--seed', type=int, required=True, help='baseline seed whose records are imported')
    ap.add_argument('--source-seed-dir', required=True, help='baseline run directory of that seed (holds reactions/<rid>/checkpoint.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--store', default=str(ROOT / 'slab_store'))
    args = ap.parse_args()
    element, facet = args.surface.split(':')
    out = Path(args.out) / f'{element}_{facet}'
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / 'log.txt').open('a')

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    recipes = json.loads((ROOT / 'campaigns/ni_drm/config/species.json').read_text())
    a, src = slab_store.model_lattice(element, args.store, log=log)
    spec = slab_store.spec(element, facet, a=a, lattice_source=src, fixed_layers=None)
    record = slab_store.get_or_create(args.store, spec, log=log)
    (out / 'slab_record_used.json').write_text(json.dumps(
        {**{k: record[k] for k in ('name', 'key_sha256', 'spec', 'relaxation', 'model', 'versions', 'reused')},
         'store': str(Path(args.store).resolve())}, indent=1))
    results = {}
    name = args.species
    try:
        source_root, entry, checkpoint_sha = resolve_records(args.source_seed_dir, args.reaction)
        summary = import_records(name, recipes[name], record, out / name, source_root,
                                 reaction=args.reaction, seed=args.seed, log=log)
        manifest_path = out / name / 'external_import.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(source_seed_dir=str(Path(args.source_seed_dir).resolve()), source_checkpoint_sha256=checkpoint_sha,
                        source_endpoint_artifact_hashes=entry['artifact_hashes'])
        manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
        results[name] = summary['best']
    except Exception as exc:
        log(f'[{name}] ERROR {type(exc).__name__}: {exc}')
        (out / name).mkdir(exist_ok=True)
        (out / name / 'error.txt').write_text(traceback.format_exc())
        results[name] = {'error': f'{type(exc).__name__}: {exc}'}
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str))
    log('ALL DONE')
    logf.close()
    if 'error' in results[name]:
        sys.exit(1)


if __name__ == '__main__':
    main()
