import os
for _k in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS'
):
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
from chavis.inputs.adsorbate_search import search
from chavis.runtime.adsorption_settings import parse_azimuth_step
PARENTS = [
    'CH4',
    'CH3',
    'CH2',
    'CH',
    'CO2',
    'CHO',
    'CO',
    'H2O',
    'OH',
    'H2',
    'CHOH',
    'COH'
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surface', required=True, help='Element:facet, e.g. Ni:fcc111')
    ap.add_argument('--species', default=','.join(PARENTS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--store', default=str(ROOT / 'slab_store'))
    ap.add_argument(
        '--adsorption-azimuth-step-deg',
        type=parse_azimuth_step,
        default='auto',
        help='adsorption rotation grid: auto (legacy slab-dependent <=45 deg) or an exact divisor of 360, e.g. 120'
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='debug: first N candidates only'
    )
    ap.add_argument(
        '--mode',
        choices=('general', 'fast'),
        default='general',
        help='height rule: general (1.6,1.0,1.3,...) or fast (1.0 then up)'
    )
    ap.add_argument(
        '--fix-layers',
        default='all',
        help='metal layers to fix: all | none | e.g. 3,4 (1 = top layer)'
    )
    ap.add_argument(
        '--slab-fix-layers',
        default=None,
        help='wf1: layers fixed while relaxing the clean slab, layer numbers (1 = top), default 3,4'
    )
    ap.add_argument(
        '--lattice-a',
        type=float,
        default=None,
        help='fcc lattice constant in A (default: optimised with the slab model/task; experimental = note only)'
    )
    ap.add_argument(
        '--lattice-source',
        default=None,
        help='where --lattice-a comes from (recorded with the slab)'
    )
    ap.add_argument(
        '--contact-scale',
        type=float,
        default=None,
        help='contact distance = scale x covalent radii sum (default 1.0)'
    )
    args = ap.parse_args()
    if args.lattice_a is not None and (not args.lattice_source or not args.lattice_source.strip()):
        ap.error('--lattice-a needs --lattice-source')
    if args.lattice_source is not None and args.lattice_a is None:
        ap.error('--lattice-source requires --lattice-a')
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
    if args.lattice_a is not None:
        a, src = (args.lattice_a, args.lattice_source)
    else:
        a, src = slab_store.model_lattice(element, args.store, log=log)
    spec = slab_store.spec(
        element,
        facet,
        a=a,
        lattice_source=src,
        fixed_layers=args.slab_fix_layers
    )
    record = slab_store.get_or_create(args.store, spec, log=log)
    (out / 'slab_record_used.json').write_text(json.dumps(
        {
            **{k: record[k] for k in (
                'name',
                'key_sha256',
                'spec',
                'relaxation',
                'model',
                'versions',
                'reused'
            )},
            'store': str(Path(args.store).resolve())
        },
        indent=1
    ))
    results = {}
    for name in args.species.split(','):
        try:
            results[name] = search(
                name,
                recipes[name],
                record,
                out / name,
                log=log,
                limit=args.limit,
                contact_scale=args.contact_scale,
                mode=args.mode,
                fix_layers=args.fix_layers,
                azimuth_step_deg=args.adsorption_azimuth_step_deg
            )['best']
        except Exception as exc:
            log(f'[{name}] ERROR {type(exc).__name__}: {exc}')
            (out / name).mkdir(exist_ok=True)
            (out / name / 'error.txt').write_text(traceback.format_exc())
            results[name] = {'error': f'{type(exc).__name__}: {exc}'}
        (out / 'results.json').write_text(json.dumps(results, indent=1, default=str))
    log('ALL DONE')
    logf.close()


if __name__ == '__main__':
    main()
