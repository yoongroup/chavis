from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes


def fixed_indices(atoms):
    if any((not isinstance(c, FixAtoms) for c in atoms.constraints)):
        raise ValueError('Unsupported constraint: explicitly restore the original mask before serialization')
    return sorted({int(i) for c in atoms.constraints for i in c.get_indices()})


def load(value):
    n = len(value['symbols'])
    ids = value['atom_ids']
    if len(ids) != n or len(set(ids)) != n or any((type(i) is not int for i in ids)):
        raise ValueError('Unique original integer atom IDs required')
    if value.get('meta', {}).get('unsupported_constraints'):
        raise ValueError('Unsupported saved constraints')
    atoms = Atoms(
        value['symbols'],
        positions=value['positions'],
        cell=value['cell'],
        pbc=value['pbc']
    )
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError('Nonfinite structure')
    atoms.set_array('rw_id', np.array(ids, dtype=int))
    atoms.info = deepcopy(value.get('meta', {}))
    ads = value.get('adsorbate_indices', atoms.info.get('adsorbate_indices'))
    if not ads or len(set(ads)) != len(ads) or any((type(i) is not int or not 0 <= i < n for i in ads)):
        raise ValueError('Valid adsorbate indices required')
    atoms.info['adsorbate_indices'] = list(ads)
    fixed = value.get('fixed_indices', [])
    if len(set(fixed)) != len(fixed) or any((type(i) is not int or not 0 <= i < n for i in fixed)):
        raise ValueError('Invalid original fixed mask')
    atoms.set_constraint(FixAtoms(indices=fixed))
    for key, setter in (
        ('initial_magmoms', atoms.set_initial_magnetic_moments),
        ('initial_charges', atoms.set_initial_charges),
        ('tags', atoms.set_tags)
    ):
        if key in value:
            arr = np.asarray(value[key])
            if len(arr) != n or not np.isfinite(arr).all():
                raise ValueError(f'Invalid {key}')
            if key == 'tags' and (arr.ndim != 1 or not np.equal(arr, np.round(arr)).all()):
                raise ValueError('Integer tags required; do not silently truncate')
            setter(arr)
    atoms.info['_automation_input_extra'] = deepcopy({
        k: v for k,
        v in value.items() if k not in {
            'symbols',
            'positions',
            'cell',
            'pbc',
            'atom_ids',
            'fixed_indices',
            'meta',
            'adsorbate_indices',
            'initial_magmoms',
            'initial_charges',
            'tags',
            'energy'
        }
    })
    return atoms


def dump(atoms):
    meta = deepcopy(atoms.info)
    extra = meta.pop('_automation_input_extra', {})
    result = dict(
        extra,
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.positions.tolist(),
        cell=atoms.cell.array.tolist(),
        pbc=atoms.pbc.tolist(),
        atom_ids=atoms.arrays['rw_id'].tolist(),
        fixed_indices=fixed_indices(atoms),
        adsorbate_indices=list(meta['adsorbate_indices']),
        meta=meta
    )
    for key in ('initial_magmoms', 'initial_charges', 'tags'):
        if key in atoms.arrays:
            result[key] = atoms.arrays[key].tolist()
    return result


def assert_compatible(first, second):
    a, b = (dump(first), dump(second))
    for key in (
        'symbols',
        'atom_ids',
        'cell',
        'pbc',
        'fixed_indices',
        'adsorbate_indices',
        'initial_magmoms',
        'initial_charges',
        'tags'
    ):
        if a.get(key) != b.get(key):
            raise ValueError(f'Endpoint mismatch: {key}')
    for key in ('charge', 'spin'):
        if (key in a['meta']) != (key in b['meta']) or a['meta'].get(key) != b['meta'].get(key):
            raise ValueError(f'Endpoint mismatch or missing metadata: {key}')
    if not np.array_equal(
        first.positions[fixed_indices(first)],
        second.positions[fixed_indices(second)]
    ):
        raise ValueError('Endpoint fixed-atom coordinates differ')
    if first.pbc.tolist() != [True, True, False]:
        raise ValueError('This adapter is validated only for slab xy periodic / z nonperiodic')


class ModelViewCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, inner, original, check=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.original = original.copy()
        self.mask = fixed_indices(original)
        self.evaluations = 0
        self.check = check or (lambda: None)

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.check()
        if atoms.get_chemical_symbols() != self.original.get_chemical_symbols() or not np.array_equal(
            atoms.arrays['rw_id'],
            self.original.arrays['rw_id']
        ):
            raise ValueError('Model composition/ID mismatch')
        if not np.array_equal(atoms.cell, self.original.cell) or not np.array_equal(atoms.pbc, self.original.pbc):
            raise ValueError('Fixed model cell/PBC changed')
        view = self.original.copy()
        view.info = deepcopy(self.original.info)
        view.positions[:] = atoms.positions
        view.set_cell(atoms.cell)
        view.set_pbc(atoms.pbc)
        view.set_constraint(FixAtoms(indices=self.mask))
        view.calc = self.inner
        energy = float(view.get_potential_energy())
        forces = np.asarray(view.get_forces(apply_constraint=False), dtype=np.float64)
        if not np.isfinite(energy) or forces.shape != (len(view), 3) or (not np.isfinite(forces).all()):
            raise ValueError('Nonfinite/invalid model response')
        self.evaluations += 1
        self.results = dict(energy=energy, forces=forces.copy())
