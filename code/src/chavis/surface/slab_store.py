from __future__ import annotations
import hashlib
import json
import math
import platform
import time
from pathlib import Path
EXPERIMENTAL_LATTICE = {
    'Ni': (
        3.52,
        'experimental fcc lattice constant (room temperature); reference value'
    ),
    'Cu': (
        3.61,
        'experimental fcc lattice constant (room temperature); reference value'
    )
}
LATTICE = EXPERIMENTAL_LATTICE
FACETS = ('fcc111', 'fcc100')
DEFAULT = {
    'size': [3, 3, 4],
    'vacuum': 10.0,
    'fixed_layers': 2,
    'relax': {
        'optimizer': 'ase.optimize.BFGS',
        'maxstep_A': 0.15,
        'fmax_eV_A': 0.01,
        'max_steps': 500
    },
    'model': 'uma-s-1p2p1',
    'task': 'oc20'
}
EOS = {
    'span': 0.04,
    'points': 25,
    'fit': 'ase.eos birchmurnaghan',
    'cell': 'fcc primitive bulk'
}


def _validate_lattice_record(rec, element, model, task):
    import numpy as np
    if any((
        rec.get(k) != v for k,
        v in {'element': element, 'model': model, 'task': task, 'eos': EOS}.items()
    )):
        raise ValueError('element/model/task/EOS settings do not match')
    a0, source = (float(rec['a_A']), rec['lattice_source'])
    scan_a = np.asarray(rec['scan_a_A'], dtype=float)
    scan_e = np.asarray(rec['scan_energy_eV'], dtype=float)
    fit = rec['fit']
    v0, e0, b0 = [float(fit[k]) for k in ('v0_A3', 'e0_eV', 'b0_eV_A3')]
    if scan_a.shape != (EOS['points'],) or scan_e.shape != scan_a.shape or (not np.all(np.isfinite(scan_a))) or (not np.all(np.isfinite(scan_e))) or (not np.all(np.diff(scan_a) > 0)) or (not np.all(scan_a > 0)):
        raise ValueError('invalid energy-volume scan')
    if not np.all(np.isfinite([a0, v0, e0, b0])) or v0 <= 0 or b0 <= 0 or (not scan_a[0] < a0 < scan_a[-1]) or (not np.isclose(
        a0,
        (4 * v0) ** (1 / 3),
        rtol=0,
        atol=5e-05
    )):
        raise ValueError('invalid equilibrium lattice fit')
    center = (scan_a[0] + scan_a[-1]) / 2
    if not np.isclose(
        (scan_a[-1] - scan_a[0]) / (2 * center),
        EOS['span'],
        rtol=0,
        atol=1e-05
    ) or not np.allclose(
        scan_a,
        np.linspace(scan_a[0], scan_a[-1], EOS['points']),
        rtol=0,
        atol=1e-05
    ):
        raise ValueError('scan does not match the requested range/grid')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('lattice_source must be recorded')
    return (a0, source)


def model_lattice(
    element,
    store,
    factory=None,
    *,
    model=None,
    task=None,
    calc=None,
    log=print
):
    import numpy as np
    model = model or DEFAULT['model']
    task = task or DEFAULT['task']
    if calc is None and (model != 'uma-s-1p2p1' or task != DEFAULT['task']):
        raise ValueError('configured slab uses uma-s-1p2p1/oc20; supply calc= for another model/task')
    path = Path(store) / 'lattice' / f'{element}_{model}_{task}.json'
    if path.is_file():
        rec = json.loads(path.read_text())
        try:
            return _validate_lattice_record(rec, element, model, task)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'stored lattice {path} is invalid; refusing to reuse: {exc}') from exc
    from ase.build import bulk
    from ase.eos import EquationOfState
    exp = EXPERIMENTAL_LATTICE.get(element)
    a_guess = exp[0] if exp else float(bulk(element).cell.cellpar()[0] * 2 ** 0.5)
    if calc is None:
        from fairchem.core import FAIRChemCalculator
        if factory is None:
            from automation_search.uma_model import ModelFactory
            factory = ModelFactory(lambda: None)
        calc = FAIRChemCalculator(factory.predictor, task_name=task)
    a_s = np.linspace(
        a_guess * (1 - EOS['span']),
        a_guess * (1 + EOS['span']),
        EOS['points']
    )
    vols, ens = ([], [])
    for a in a_s:
        at = bulk(element, 'fcc', a=float(a))
        at.calc = calc
        vols.append(at.get_volume())
        ens.append(float(at.get_potential_energy()))
    v0, e0, b0 = EquationOfState(vols, ens, eos='birchmurnaghan').fit()
    a0 = round(float((4.0 * v0) ** (1.0 / 3.0)), 4)
    if not a_s[0] < a0 < a_s[-1]:
        raise RuntimeError(f'{element} {model}/{task}: fitted a={a0:.4f} A outside the scanned range')
    source = f"{model} {task} equilibrium fcc lattice constant (EOS fit, {EOS['points']} points, +-{EOS['span'] * 100:.0f}%); experimental value is a note only" + (f' ({exp[0]} A, {100 * (exp[0] - a0) / a0:+.2f}% vs model)' if exp else '')
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        'element': element,
        'model': model,
        'task': task,
        'a_A': a0,
        'lattice_source': source,
        'eos': EOS,
        'scan_a_A': a_s.round(5).tolist(),
        'scan_energy_eV': ens,
        'fit': {'v0_A3': float(v0), 'e0_eV': float(e0), 'b0_eV_A3': float(b0)},
        'experimental_note': {'a_A': exp[0], 'source': exp[1]} if exp else None,
        'versions': versions(),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    _validate_lattice_record(rec, element, model, task)
    path.write_text(json.dumps(rec, indent=1))
    log(f"[lattice] {element} {model}/{task}: a0 = {a0:.4f} A (experimental note {(exp[0] if exp else '-')})")
    return (a0, source)


def spec(
    element,
    facet,
    *,
    a=None,
    lattice_source=None,
    size=None,
    vacuum=None,
    fixed_layers=None,
    relax=None
):
    if facet not in FACETS:
        raise ValueError(f'unsupported facet {facet!r}; supported: {FACETS}')
    if a is None:
        raise ValueError('pass a= and lattice_source= (use model_lattice for the model-optimised value; experimental values are reference values only)')
    if not isinstance(lattice_source, str) or not lattice_source.strip():
        raise ValueError('lattice_source must be recorded when a is given')
    a = float(a)
    if not math.isfinite(a) or a <= 0:
        raise ValueError('a must be a finite positive lattice constant')
    size = list(size if size is not None else DEFAULT['size'])
    return {
        'element': element,
        'facet': facet,
        'a': float(a),
        'lattice_source': lattice_source,
        'size': size,
        'vacuum': float(vacuum if vacuum is not None else DEFAULT['vacuum']),
        'fixed_layers': fixed_layer_list(fixed_layers, size[2]),
        'relax': dict(relax if relax is not None else DEFAULT['relax']),
        'model': DEFAULT['model'],
        'task': DEFAULT['task']
    }


def fixed_layer_list(fixed_layers, n_layers):
    from chavis.surface.metal_mode import parse
    if fixed_layers is None:
        n = DEFAULT['fixed_layers']
        return list(range(n_layers - n + 1, n_layers + 1))
    if isinstance(fixed_layers, int) and (not isinstance(fixed_layers, bool)):
        raise ValueError('fixed_layers takes layer numbers (1 = top), e.g. "3,4" or [3, 4]; a bare int is ambiguous')
    p = parse(fixed_layers)
    out = list(range(1, n_layers + 1)) if p == 'all' else list(p)
    if any((l > n_layers for l in out)):
        raise ValueError(f'slab has {n_layers} layers; cannot fix {out}')
    return out


def _key_fixed(s):
    fl, n = (s['fixed_layers'], s['size'][2])
    if isinstance(fl, int):
        return fl
    k = len(fl)
    return k if k and fl == list(range(n - k + 1, n + 1)) else fl


def key_of(s):
    ident = {k: s[k] for k in (
        'element',
        'facet',
        'a',
        'size',
        'vacuum',
        'relax',
        'model',
        'task'
    )}
    ident['fixed_layers'] = _key_fixed(s)
    digest = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    size = 'x'.join((str(v) for v in s['size']))
    return (f"{s['element']}_{s['facet']}_{size}_a{s['a']:.3f}_{digest[:10]}", digest)


def build_atoms(s):
    from ase.build import fcc100, fcc111
    from ase.constraints import FixAtoms
    builder = {'fcc111': fcc111, 'fcc100': fcc100}[s['facet']]
    atoms = builder(s['element'], size=tuple(s['size']), a=s['a'], vacuum=s['vacuum'])
    fl = s['fixed_layers']
    if isinstance(fl, int):
        fl = list(range(s['size'][2] - fl + 1, s['size'][2] + 1))
    fixed = [i for i, t in enumerate(atoms.get_tags()) if t in set(fl)]
    atoms.set_constraint(FixAtoms(indices=fixed))
    atoms.pbc = [True, True, False]
    return (atoms, fixed)


def versions():
    out = {'python': platform.python_version()}
    for name in ('ase', 'numpy', 'scipy', 'torch', 'fairchem.core', 'rdkit'):
        try:
            mod = __import__(name, fromlist=['__version__'])
            out[name] = getattr(mod, '__version__', 'unknown')
        except Exception as exc:
            out[name] = f'unavailable: {type(exc).__name__}'
    return out


def _structure_dict(atoms, fixed, profile, energy):
    return {
        'symbols': atoms.get_chemical_symbols(),
        'positions': atoms.positions.tolist(),
        'cell': atoms.cell.array.tolist(),
        'pbc': [bool(x) for x in atoms.pbc],
        'atom_ids': list(range(len(atoms))),
        'fixed_indices': list(fixed),
        'layers': _layers_from_tags(atoms),
        'adsorbate_indices': [],
        'energy': energy,
        'meta': {
            'atom_id_source': 'file',
            'adsorbate_indices': [],
            'surface_profile': profile,
            'charge': 0,
            'spin': 0
        }
    }


def _profile(s, atoms, fixed):
    return {
        'element': s['element'],
        'facet': s['facet'],
        'size': s['size'],
        'a': s['a'],
        'vacuum': s['vacuum'],
        'fixed_layers': _key_fixed(s),
        'fixed_layer_numbers': list(s['fixed_layers'] if not isinstance(s['fixed_layers'], int) else range(
            s['size'][2] - s['fixed_layers'] + 1,
            s['size'][2] + 1
        )),
        'fmax': s['relax']['fmax_eV_A'],
        'steps': s['relax']['max_steps'],
        'n_slab': len(atoms),
        'fixed_indices': list(fixed),
        'cell': atoms.cell.array.tolist(),
        'pbc': [True, True, False],
        'source': {
            'kind': f"ASE {s['facet']}",
            'lattice_source': s['lattice_source'],
            'cell_relaxation': False
        }
    }


def load(store, s):
    name, _ = key_of(s)
    path = Path(store) / name / 'record.json'
    return json.loads(path.read_text()) if path.is_file() else None


def get_or_create(store, s, factory=None, log=print):
    store = Path(store)
    name, digest = key_of(s)
    folder = store / name
    record_path = folder / 'record.json'
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        if record.get('key_sha256') != digest:
            raise ValueError(f'stored slab {name} has a different key; refusing to reuse')
        if not record.get('relaxation', {}).get('converged', False):
            raise RuntimeError(f'stored slab {name} did not converge; refusing to reuse (remove {folder} to rebuild)')
        log(f'[slab] reuse {name}')
        record['reused'] = True
        return record
    from ase.optimize import BFGS
    from fairchem.core import FAIRChemCalculator
    import numpy as np
    folder.mkdir(parents=True, exist_ok=True)
    if factory is None:
        from automation_search.uma_model import ModelFactory
        factory = ModelFactory(lambda: None)
    atoms, fixed = build_atoms(s)
    atoms.calc = FAIRChemCalculator(factory.predictor, task_name=s['task'])
    relax = s['relax']
    t0 = time.monotonic()
    c0 = getattr(factory, 'calls', None)
    opt = BFGS(
        atoms,
        maxstep=relax['maxstep_A'],
        logfile=str(folder / 'relax.log'),
        trajectory=str(folder / 'relax.traj')
    )
    ok = bool(opt.run(fmax=relax['fmax_eV_A'], steps=relax['max_steps']))
    energy = float(atoms.get_potential_energy())
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    profile = _profile(s, atoms, fixed)
    structure = _structure_dict(atoms, fixed, profile, energy)
    assets = getattr(factory, 'assets', {})
    record = {
        'schema': 'chavis_ver2_slab_record_v1',
        'name': name,
        'key_sha256': digest,
        'spec': s,
        'structure': structure,
        'profile': profile,
        'relaxation': {
            'converged': ok,
            'steps': int(opt.nsteps),
            'final_fmax_eV_A': fmax,
            'energy_eV': energy,
            'model_calls': factory.calls - c0 if c0 is not None else None,
            'wall_seconds': time.monotonic() - t0,
            'optimizer': relax['optimizer'],
            'maxstep_A': relax['maxstep_A'],
            'log': 'relax.log',
            'trajectory': 'relax.traj'
        },
        'model': {
            'name': s['model'],
            'task': s['task'],
            'revision': 'f611b917d9c68566bbbeccbb0aa0f7cad1696cb2',
            'assets': {
                k: {'sha256': v.get('sha256'), 'bytes': v.get('bytes')} for k,
                v in assets.items()
            }
        },
        'versions': versions(),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'reused': False
    }
    atoms.calc = None
    record_path.write_text(json.dumps(record, indent=1))
    log(f'[slab] built {name}: converged={ok} steps={opt.nsteps} fmax={fmax:.4f}')
    if not ok:
        raise RuntimeError(f'slab {name} did not converge; record kept at {record_path}')
    return record


def record_for_run(surf_dir, default_store):
    used = json.loads((Path(surf_dir) / 'slab_record_used.json').read_text())
    store = Path(used.get('store') or default_store)
    return json.loads((store / used['name'] / 'record.json').read_text())


def to_atoms(record):
    from ase import Atoms
    from ase.constraints import FixAtoms
    st = record['structure']
    a = Atoms(st['symbols'], positions=st['positions'], cell=st['cell'], pbc=st['pbc'])
    a.set_constraint(FixAtoms(indices=st['fixed_indices']))
    return a


def _layers_from_tags(atoms):
    return [int(t) for t in atoms.get_tags()]


def backfill_layers(store):
    import json
    from pathlib import Path
    from chavis.surface.metal_mode import layers
    done = []
    for path in Path(store).glob('*/record.json'):
        rec = json.loads(path.read_text())
        if not rec['structure'].get('layers'):
            rec['structure']['layers'] = layers(rec)
            rec['structure'].setdefault('meta', {})['layers_note'] = '1 = top layer; backfilled from heights'
            path.write_text(json.dumps(rec, indent=1))
            done.append(path.parent.name)
    return done
