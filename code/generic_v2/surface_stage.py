"""Isolated clean-surface preparation, with actual EOS/slab model-call accounting."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'support/runtime_support')]

from generic_v2.common import atomic_json, read_json, sha256
from generic_v2.surfaces import inspect_cache, surface_definition


def prepare(source_store, target_store, surface, expected, counter, phases):
    """Copy only verified cache bytes, then run the unchanged slab-store API.

    ``counter`` observes the aggregate actual predictor calls across all model
    factories. A failed phase is recorded in ``finally`` and stays chargeable.
    """
    from chavis.surface import slab_store
    source_store, target_store = Path(source_store), Path(target_store)
    before = inspect_cache(source_store, surface)
    if before != expected:
        raise ValueError('Surface source cache changed since request fingerprint was frozen')
    target_store.mkdir(parents=True, exist_ok=False)
    for rel, digest in before['files'].items():
        src, dst = source_store / rel, target_store / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256(dst) != digest:
            raise RuntimeError(f'Cache copy hash mismatch: {rel}')

    @contextmanager
    def phase(name):
        start, calls = time.monotonic(), counter()
        status = 'complete'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            phases[name] = {'calls': counter() - calls, 'seconds': time.monotonic() - start, 'status': status}

    definition = surface_definition(surface)
    with phase('eos'):
        a, source = slab_store.model_lattice(definition['element'], target_store)
    # Predictor wrappers close over their factory. Collect the now-unreachable
    # EOS factory cycle before the unchanged slab API creates its own factory,
    # preserving the original model initialization/RNG sequence without overlap.
    gc.collect()
    spec = slab_store.spec(definition['element'], definition['facet'], a=a, lattice_source=source)
    with phase('slab'):
        record = slab_store.get_or_create(target_store, spec)
    after = inspect_cache(target_store, surface)
    if not after['slab_cached']:
        raise RuntimeError('Surface preparation did not produce a validated slab cache')
    return {'schema': 'chavis_surface_preparation_v1', 'surface': definition,
            'source_store': str(source_store.resolve()), 'store': str(target_store.resolve()),
            'source_cache': before, 'prepared_cache': after, 'record_reused': record['reused'],
            'phase_costs': phases,
            'accounting': 'Actual new predictor calls; reusable historical lattice/slab preparation is not recharged'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface', required=True)
    p.add_argument('--source-store', required=True)
    p.add_argument('--store', required=True)
    p.add_argument('--expected-cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cost-file', required=True)
    args = p.parse_args(argv)
    # Merely importing the factory module does not create or load a model.
    import automation_search.uma_model as uma
    count = {'calls': 0, 'models': 0}
    original = uma.ModelFactory

    class CountingFactory(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            count['models'] += 1
            inner = self.predictor.predict

            def counted(*pa, **pk):
                before = self.calls
                try:
                    return inner(*pa, **pk)
                finally:
                    count['calls'] += self.calls - before
            self.predictor.predict = counted

    uma.ModelFactory = CountingFactory
    start, error, phases = time.monotonic(), None, {}
    try:
        result = prepare(args.source_store, args.store, args.surface, read_json(args.expected_cache),
                         lambda: count['calls'], phases)
        if sum(row['calls'] for row in phases.values()) != count['calls']:
            raise RuntimeError('Surface phase counters do not reconcile to total model calls')
        atomic_json(args.out, result)
    except BaseException as exc:
        error = repr(exc)
    finally:
        uma.ModelFactory = original
        atomic_json(args.cost_file, {'tool': 'generic_v2.surface_stage', 'calls': count['calls'],
                    'seconds': time.monotonic() - start, 'models_created': count['models'],
                    'phase_costs': phases, 'surface': args.surface, 'error': error})
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
