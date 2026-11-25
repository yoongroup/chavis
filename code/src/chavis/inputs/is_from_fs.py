from __future__ import annotations
import time
import numpy as np
SCAN = {
    'step_A': 0.1,
    'optimizer': 'ase.optimize.BFGS',
    'maxstep_A': 0.1,
    'fmax_eV_A': 0.05,
    'max_steps': 300,
    'source': 'automation_stages.bond_scan settings'
}


def _float64(calc):
    from ase.calculators.calculator import Calculator, all_changes

    class Float64(Calculator):
        implemented_properties = ['energy', 'forces']

        def calculate(
            self,
            atoms=None,
            properties=('energy',),
            system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            a = self.atoms.copy()
            a.set_constraint()
            a.calc = calc
            self.results = {
                'energy': float(a.get_potential_energy()),
                'forces': np.asarray(a.get_forces(), dtype=np.float64)
            }
            a.calc = None
    return Float64()


def fragment(edges, start, cut):
    g = {}
    for a, b in edges:
        if {a, b} == set(cut):
            continue
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    seen, todo = ({start}, [start])
    while todo:
        for j in g.get(todo.pop(), ()):
            if j not in seen:
                seen.add(j)
                todo.append(j)
    return sorted(seen)


def derive_is(sp, fs_row, fix_layers, log=print):
    from ase.constraints import FixAtoms, FixBondLengths
    from ase.optimize import BFGS
    from chavis.search.planning import scan_targets
    from chavis.surface.metal_mode import fixed_indices
    a, b = fs_row['bond_lost']
    ref = {tuple(sorted(e)) for e in sp.ref}
    d_ref = float(sp.rl[tuple(sorted((a, b)))])
    ads_idx = list(range(sp.n, sp.n + len(sp.sym)))
    fa, fb = (fragment(ref, a, (a, b)), fragment(ref, b, (a, b)))
    anchor, lead, moving = (a, b, fb) if len(fb) <= len(fa) else (b, a, fa)
    atoms = sp.atoms(
        fs_row['final_slab_positions'],
        fs_row['final_adsorbate_positions']
    )
    from ase.geometry import find_mic
    pos = atoms.positions
    pos[lead] = pos[anchor] + find_mic(
        pos[lead] - pos[anchor],
        atoms.cell,
        atoms.pbc
    )[0]
    for j in moving:
        if j != lead:
            pos[j] = pos[lead] + find_mic(pos[j] - pos[lead], atoms.cell, atoms.pbc)[0]
    atoms.positions = pos
    fixed = fixed_indices(sp.record, fix_layers)
    d_fs = float(atoms.get_distance(anchor, lead, mic=True))
    targets = scan_targets(d_fs, d_ref) if d_fs > d_ref else [d_ref]
    calc = _float64(sp.ensure_calc())
    t0 = time.monotonic()
    c0 = sp.factory.calls
    points = []
    for k, target in enumerate(targets):
        atoms.set_constraint()
        v = atoms.get_distance(anchor, lead, mic=True, vector=True)
        shift = (target - np.linalg.norm(v)) * v / np.linalg.norm(v)
        atoms.positions[moving] += shift
        atoms.set_constraint([
            FixAtoms(indices=fixed),
            FixBondLengths([[anchor, lead]])
        ])
        atoms.calc = calc
        opt = BFGS(atoms, maxstep=SCAN['maxstep_A'], logfile=None)
        ok = bool(opt.run(fmax=SCAN['fmax_eV_A'], steps=SCAN['max_steps']))
        points.append({
            'k': k,
            'd_target_A': round(target, 4),
            'bond_A': round(float(atoms.get_distance(anchor, lead, mic=True)), 4),
            'energy_eV': float(atoms.get_potential_energy()),
            'converged': ok,
            'steps': int(opt.nsteps)
        })
    atoms.calc = None
    atoms.set_constraint()
    res = sp.relax(atoms, fix_layers)
    return {
        'anchor': anchor,
        'lead': lead,
        'moving': moving,
        'd_fs_A': round(d_fs, 4),
        'd_gas_A': round(d_ref, 4),
        'scan_settings': SCAN,
        'scan': points,
        'scan_model_calls': sp.factory.calls - c0 - res['model_calls'],
        'scan_seconds': round(time.monotonic() - t0 - res['seconds'], 2),
        **res
    }
