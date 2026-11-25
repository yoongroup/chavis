from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
REFINE = {
    'optimizer': 'ase.optimize.BFGS',
    'maxstep_A': 0.15,
    'fmax_eV_A': 0.01,
    'max_steps': 500,
    'task': 'oc20',
    'fix_layers_default': '3,4'
}


def transition(before, after):
    from chavis.inputs.minima import form, _outcome
    ob, oa = (_outcome(before), _outcome(after))
    fb, fa = (form(before), form(after))
    pb = np.asarray(before['final_adsorbate_positions'], float)
    pa = np.asarray(after['final_adsorbate_positions'], float)
    return {
        'outcome': f'{ob}->{oa}',
        'form': f'{fb}->{fa}',
        'outcome_changed': ob != oa,
        'form_changed': fb != fa,
        'max_adsorbate_move_A': round(float(np.linalg.norm(pa - pb, axis=1).max()), 4),
        'energy_change_eV': round(
            after['energy_eV'] - before['energy_eV'],
            4
        ) if after.get('energy_eV') is not None else None
    }


def refine_species(species_dir, record, minima_to_refine, fix_layers='3,4', log=print):
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from fairchem.core import FAIRChemCalculator
    from automation_search.uma_model import ModelFactory
    from chavis.inputs.adsorbate_search import _site_representatives, outcome, gpu_guard
    from chavis.judgment.structure_labels import label, adsorbate_graph, reference_lengths
    from chavis.surface import slab_store
    from chavis.surface.metal_mode import fixed_indices, describe
    species_dir = Path(species_dir)
    rows = {r['index']: r for r in (json.loads(l) for l in (species_dir / 'candidates.jsonl').open())}
    mol = json.loads((species_dir / 'gas' / 'molecule.json').read_text())
    sym = mol['symbols']
    slab = slab_store.to_atoms(record)
    n = len(slab)
    clean = slab.positions.copy()
    gas = Atoms(sym, positions=mol['positions_after_placeholder_removal'])
    ge = adsorbate_graph(gas, list(range(len(sym))))
    ref = {(a + n, b + n) for a, b in ge}
    rl = {(a + n, b + n): v for (a, b), v in reference_lengths(gas, ge).items()}
    _, sites, _ = _site_representatives(record['structure'])
    fixed = fixed_indices(record, fix_layers)
    metal = describe(record, fix_layers)
    normal_fixed = list(record['structure']['fixed_indices'])
    factory = ModelFactory(lambda: None)
    calc = FAIRChemCalculator(factory.predictor, task_name=REFINE['task'])
    out = []
    with (species_dir / 'refined.jsonl').open('w') as fh:
        for k, m in enumerate(minima_to_refine):
            if k % 10 == 0:
                gpu_guard(log=log)
            before = rows[m['representative']]
            a = slab.copy()
            a += Atoms(sym, positions=before['final_adsorbate_positions'])
            if before.get('final_slab_positions'):
                a.positions[:n] = before['final_slab_positions']
            a.set_constraint(FixAtoms(indices=fixed))
            a.calc = calc
            t0 = time.monotonic()
            c0 = factory.calls
            opt = BFGS(a, maxstep=REFINE['maxstep_A'], logfile=None)
            try:
                ok = bool(opt.run(fmax=REFINE['fmax_eV_A'], steps=REFINE['max_steps']))
                err = None
            except Exception as exc:
                ok, err = (False, f'{type(exc).__name__}: {exc}')
            energy = float(a.get_potential_energy()) if err is None else None
            lab = label(
                a,
                list(range(n)),
                list(range(n, len(a))),
                reference_edges=ref,
                reference_bond_lengths=rl,
                fixed_slab_indices=normal_fixed,
                clean_slab_positions=clean,
                metal_frozen=not metal['free_layers'],
                sites=sites
            )
            after = {
                'index': before['index'],
                'energy_eV': energy,
                'labels': lab,
                'outcome': outcome(lab) if err is None else 'error',
                'final_adsorbate_positions': a.positions[n:].round(5).tolist()
            }
            row = {
                'minimum_representative': m['representative'],
                'frozen_form': m['form'],
                'frozen_outcome': m['outcome'],
                'frozen_energy_eV': before['energy_eV'],
                'metal': metal,
                'settings': REFINE,
                'converged': ok,
                'error': err,
                'steps': int(opt.nsteps),
                'model_calls': factory.calls - c0,
                'seconds': round(time.monotonic() - t0, 2),
                'energy_eV': energy,
                'outcome': after['outcome'],
                'labels': lab,
                'transition': transition(before, after) if err is None else None,
                'final_slab_positions': a.positions[:n].round(5).tolist(),
                'final_adsorbate_positions': after['final_adsorbate_positions']
            }
            row['binder_bonded'] = binder_bonded(lab, n, mol.get('binder_index'))
            row['binder_metal_ratio'] = binder_metal_ratio(lab, n, mol.get('binder_index'))
            out.append(row)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            a.calc = None
    del calc, factory
    dedupe_refined(out, record, sym)
    with (species_dir / 'refined.jsonl').open('w') as fh:
        for row in out:
            fh.write(json.dumps(row) + '\n')
    return out


def binder_metal_ratio(labels, n_slab, binder_index):
    if binder_index is None:
        return None
    return labels['binding']['nearest'][str(n_slab + int(binder_index))]['ratio']


def binder_bonded(labels, n_slab, binder_index):
    if binder_index is None:
        return None
    return n_slab + int(binder_index) in {c['atom'] for c in labels['binding']['contacts']}


def upside_down(rows, ratio_scale, min_above_eV):
    es = [r['energy_eV'] for r in rows if r.get('energy_eV') is not None and r.get('converged', True) and (r.get('outcome') not in ('dissociated', 'error'))]
    if not es:
        return set()
    e0 = min(es)
    return {id(r) for r in rows if r.get('binder_metal_ratio') is not None and r['binder_metal_ratio'] > ratio_scale and (r.get('energy_eV') is not None) and (r['energy_eV'] - e0 > min_above_eV)}


def intercalated(row):
    return bool(((row.get('labels') or {}).get('intercalation') or {}).get('flag'))


def is_candidates(
    refined,
    outcomes,
    is_per_reaction='all',
    require_binder_bonded=True,
    binder_ratio_scale=1.3,
    binder_min_above_eV=1.0,
    exclude_intercalated=True
):
    skip = upside_down(
        refined,
        binder_ratio_scale,
        binder_min_above_eV
    ) if require_binder_bonded else set()
    keep = sorted(
        (r for r in refined if r['outcome'] in outcomes and r['energy_eV'] is not None and r.get('is_representative', True) and r.get('converged', True) and (id(r) not in skip) and (not (exclude_intercalated and intercalated(r)))),
        key=lambda r: r['energy_eV']
    )
    if is_per_reaction not in ('all', None):
        keep = keep[:int(is_per_reaction)]
    return keep


SAME_ADS_A = 0.1
SAME_SLAB_A = 0.1


def dedupe_refined(refined, record, symbols):
    from chavis.surface import slab_store
    from chavis.surface.symmetry import surface_operations, equivalent_with_slab
    slab = slab_store.to_atoms(record)
    ops = surface_operations(slab)
    cell = slab.cell.array
    slab_sym = slab.get_chemical_symbols()
    reps = []
    for r in sorted(
        refined,
        key=lambda x: (
            x['energy_eV'] is None,
            x['energy_eV'] or 0.0,
            x['minimum_representative']
        )
    ):
        r['merged_with'] = []
        if r['energy_eV'] is None or not r.get('converged', True):
            r['refined_group'] = None
            r['is_representative'] = False
            continue
        for rep in reps:
            if rep['outcome'] == r['outcome'] and abs(rep['energy_eV'] - r['energy_eV']) < 0.05 and equivalent_with_slab(
                rep['final_adsorbate_positions'],
                r['final_adsorbate_positions'],
                symbols,
                rep['final_slab_positions'],
                r['final_slab_positions'],
                slab_sym,
                ops,
                cell,
                SAME_ADS_A,
                SAME_SLAB_A
            ):
                r['refined_group'] = rep['refined_group']
                r['is_representative'] = False
                rep['merged_with'].append(r['minimum_representative'])
                break
        else:
            r['refined_group'] = len(reps)
            r['is_representative'] = True
            reps.append(r)
    return refined
