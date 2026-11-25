"""CPU-only surface identity and cache validation; never creates a model."""
from __future__ import annotations

import math
from pathlib import Path

from .common import REVISION, read_json, sha256

DEFAULT_SURFACE = 'Ni:fcc111'
SURFACES = ('Ni:fcc111', 'Ni:fcc100', 'Cu:fcc111', 'Rh:fcc111')


def surface_definition(value=DEFAULT_SURFACE):
    if value not in SURFACES:
        raise ValueError(f'Unsupported surface {value!r}; choose one of {SURFACES}')
    element, facet = value.split(':')
    return {'argument': value, 'key': f'{element}_{facet}', 'element': element, 'facet': facet}


def _validate_record(record, spec):
    from chavis.surface import slab_store
    import numpy as np
    name, digest = slab_store.key_of(spec)
    if record.get('name') != name or record.get('key_sha256') != digest:
        raise ValueError('slab identity/hash does not match the requested surface specification')
    actual = dict(record.get('spec') or {})
    # Historical cache keys used a bottom-layer count, equivalent to [3, 4].
    if isinstance(actual.get('fixed_layers'), int):
        n, count = actual['size'][2], actual['fixed_layers']
        actual['fixed_layers'] = list(range(n - count + 1, n + 1))
    if actual != spec:
        raise ValueError('slab specification does not match the requested surface')
    model = record.get('model') or {}
    if (model.get('name'), model.get('task'), model.get('revision')) != (spec['model'], spec['task'], REVISION):
        raise ValueError('slab model/task/revision does not match the pinned model')
    from support.runtime_support.automation_search.uma_model import ASSETS
    assets = model.get('assets') or {}
    if {name: (assets.get(name) or {}).get('sha256') for name in ASSETS} != ASSETS:
        raise ValueError('slab model asset hashes do not match the pinned model')
    relax = record.get('relaxation') or {}
    if relax.get('converged') is not True:
        raise ValueError('slab relaxation did not converge')
    force = relax.get('final_fmax_eV_A')
    if not isinstance(force, (int, float)) or isinstance(force, bool) or not math.isfinite(force) or force < 0 or force > spec['relax']['fmax_eV_A'] + 1e-12:
        raise ValueError('slab final force fails its requested convergence tolerance')
    st = record.get('structure') or {}
    n = math.prod(spec['size'])
    if st.get('symbols') != [spec['element']] * n:
        raise ValueError('slab structure element/count does not match the surface')
    xyz, cell = np.asarray(st.get('positions'), dtype=float), np.asarray(st.get('cell'), dtype=float)
    if xyz.shape != (n, 3) or cell.shape != (3, 3) or not np.isfinite(xyz).all() or not np.isfinite(cell).all():
        raise ValueError('slab coordinates/cell are invalid')
    if st.get('pbc') != [True, True, False] or np.linalg.det(cell) <= 0:
        raise ValueError('slab periodicity/cell is invalid')
    atoms, fixed = slab_store.build_atoms(spec)
    if st.get('fixed_indices') != fixed or not np.allclose(cell, atoms.cell.array, atol=1e-10, rtol=0):
        raise ValueError('slab fixed atoms/cell differ from the requested ASE slab')
    if not np.allclose(xyz[fixed], atoms.positions[fixed], atol=1e-10, rtol=0):
        raise ValueError('fixed slab coordinates differ from the requested ASE slab')
    profile = record.get('profile') or {}
    if profile.get('element') != spec['element'] or profile.get('facet') != spec['facet']:
        raise ValueError('slab profile does not identify the requested surface')
    return name


def inspect_cache(store, surface=DEFAULT_SURFACE):
    """Validate reusable bytes or report the work needed at the first GPU run.

    Absence is allowed; a present invalid cache is an error, never silently
    substituted with another surface or the experimental lattice constant.
    """
    from chavis.surface import slab_store
    definition = surface_definition(surface)
    store = Path(store)
    lattice = store / 'lattice' / f"{definition['element']}_{slab_store.DEFAULT['model']}_{slab_store.DEFAULT['task']}.json"
    result = {**definition, 'lattice_cached': lattice.is_file(), 'slab_cached': False,
              'files': {}, 'eos_points_if_missing': slab_store.EOS['points'],
              'model_revision': REVISION}
    if not lattice.is_file():
        result['required_work'] = ['model_lattice_EOS', 'clean_slab_relaxation']
        return result
    a, source = slab_store._validate_lattice_record(
        read_json(lattice), definition['element'], slab_store.DEFAULT['model'], slab_store.DEFAULT['task'])
    spec = slab_store.spec(definition['element'], definition['facet'], a=a, lattice_source=source)
    name, digest = slab_store.key_of(spec)
    record = store / name / 'record.json'
    result.update(lattice_a_A=a, slab_name=name, slab_key_sha256=digest, spec=spec)
    result['files'][lattice.relative_to(store).as_posix()] = sha256(lattice)
    if record.is_file():
        _validate_record(read_json(record), spec)
        result['slab_cached'] = True
        result['files'][record.relative_to(store).as_posix()] = sha256(record)
    result['required_work'] = [] if result['slab_cached'] else ['clean_slab_relaxation']
    return result
