from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path
import numpy as np
SEARCH = {
    'optimizer': 'ase.optimize.BFGS',
    'maxstep_A': 0.15,
    'fmax_eV_A': 0.05,
    'max_steps': 300,
    'metal': 'fix_layers option (default all)',
    'task': 'oc20',
    'contact_scale': 1.0,
    'azimuth_step_deg': 'auto (<= 45, closed under slab rotations)',
    'dedupe_rmsd_A': 0.1,
    'dedupe': 'surface symmetry operations of the slab (spglib, symprec 0.01 A)',
    'height_ladder': {
        'ends': [1.6, 1.0],
        'middle': 1.3,
        'refine': [1.45, 1.15],
        'rule': '1.6, 1.0, 1.3 always; refine by outcome only'
    },
    'fast_heights': {
        'sequence': [1.0, 1.3, 1.6],
        'raise_if': {
            'placeholder (Cl) molecule': ['dissociated'],
            'no placeholder': ['floating', 'dissociated']
        }
    }
}


def gpu_guard(limit_c=85, resume_c=75, log=print):

    def temp():
        try:
            out = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=temperature.gpu',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=10
            ).stdout.strip().splitlines()
            return max((int(x) for x in out)) if out else None
        except Exception:
            return None
    t = temp()
    if t is not None and t >= limit_c:
        log(f'[guard] GPU {t} C >= {limit_c} C, pausing')
        while t is not None and t > resume_c:
            time.sleep(20)
            t = temp()
        log(f'[guard] GPU {t} C, resuming')
    return t


def gas_molecule(name, recipe, out_dir, log=print):
    from automation_search.uma_model import ModelFactory
    from chavis.inputs.fragment import build_fragment, finalize_direction
    from chavis.inputs.fresh_one import _gas_relax
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fragment = build_fragment(name, recipe['smiles'], recipe['mode'])
    factory = ModelFactory(lambda: None)
    gas = _gas_relax(fragment, factory, out_dir, lambda: None)
    finalize_direction(fragment)
    binder = fragment.binder if recipe['mode'] == 'Cl' else None
    rec = {
        'species': name,
        'smiles': recipe['smiles'],
        'mode': recipe['mode'],
        'charge': fragment.charge,
        'spin': fragment.spin,
        'spin_reason': fragment.spin_reason,
        'gas': gas,
        'symbols': list(fragment.symbols),
        'positions_after_placeholder_removal': [list(map(float, p)) for p in fragment.positions],
        'binder_index': binder,
        'rdkit_log': fragment.steps
    }
    (out_dir / 'molecule.json').write_text(json.dumps(rec, indent=1, default=str))
    del factory
    return (list(fragment.symbols), np.asarray(fragment.positions, float), binder, rec)


def _site_representatives(slab_structure):
    from chavis.search.surface_sites import enumerate_terrace_sites
    sites, summary = enumerate_terrace_sites(slab_structure, [])
    reps = {}
    centre = np.mean(
        np.asarray(slab_structure['cell'])[:2, :2],
        axis=0
    ) * 0 + 0.5 * (np.asarray(slab_structure['cell'][0][:2]) + np.asarray(slab_structure['cell'][1][:2]))
    for s in sites:
        d = float(np.linalg.norm(np.asarray(s['position'][:2]) - centre))
        if s['kind'] not in reps or d < reps[s['kind']][1]:
            reps[s['kind']] = (s, d)
    return ([reps[k][0] for k in sorted(reps)], sites, summary)


def _place(slab_atoms, symbols, mol_positions, site_xy, slab_structure, scale):
    from ase import Atoms
    from chavis.inputs.contact_placement import contact_target
    from chavis.search.geometry import from_ase
    n = len(slab_atoms)
    top = float(slab_atoms.positions[:, 2].max())
    mol = Atoms(
        symbols,
        positions=np.asarray(mol_positions) + [site_xy[0], site_xy[1], top + 6.0]
    )
    a = slab_atoms.copy()
    a += mol
    a.info['adsorbate_indices'] = list(range(n, len(a)))
    s = from_ase(a)
    ads = list(range(n, len(a)))
    lead = a.positions[n].tolist()
    target, contact = contact_target(s, ads, ads, lead, [0.0, 0.0, 1.0], scale)
    a.positions[n:] += np.asarray(target) - a.positions[n]
    return (a, contact)


def search(
    name,
    recipe,
    slab_record,
    out_dir,
    *,
    log=print,
    limit=None,
    contact_scale=None,
    mode='general',
    fix_layers='all',
    azimuth_step_deg='auto'
):
    from chavis.runtime.adsorption_settings import parse_azimuth_step
    requested_step = parse_azimuth_step(azimuth_step_deg)
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from fairchem.core import FAIRChemCalculator
    from automation_search.uma_model import ModelFactory
    from chavis.inputs.orientations import generate
    from chavis.judgment.structure_labels import label, adsorbate_graph, reference_lengths
    from chavis.surface.slab_store import to_atoms
    from ase import Atoms
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    settings = dict(SEARCH)
    if requested_step != 'auto':
        settings['azimuth_step_deg'] = requested_step
    if contact_scale is not None:
        settings['contact_scale'] = float(contact_scale)
    if mode not in ('general', 'fast'):
        raise ValueError(f'mode must be general or fast, not {mode!r}')
    settings['mode'] = mode
    from chavis.surface.metal_mode import fixed_indices as metal_fixed, describe as metal_describe
    settings['metal'] = metal_describe(slab_record, fix_layers)
    fixed_now = metal_fixed(slab_record, fix_layers)
    from chavis.surface.symmetry import surface_operations, equivalent, resolve_azimuth_step
    slab = to_atoms(slab_record)
    ops = surface_operations(slab)
    step = resolve_azimuth_step(requested_step, ops)
    settings['azimuth_step_used_deg'] = step
    symbols, mol, binder, mol_rec = gas_molecule(name, recipe, out_dir / 'gas', log)
    n = len(slab)
    structure = slab_record['structure']
    fixed_normal = list(structure['fixed_indices'])
    clean = slab.positions.copy()
    gas_atoms = Atoms(symbols, positions=mol)
    gas_edges = adsorbate_graph(gas_atoms, list(range(len(symbols))))
    ref_edges = {(a + n, b + n) for a, b in gas_edges}
    ref_lengths = {
        (a + n, b + n): v for (a, b),
        v in reference_lengths(gas_atoms, gas_edges).items()
    }
    reps, all_sites, site_summary = _site_representatives(structure)
    orients = generate(mol, placeholder_binder=binder, azimuth_step_deg=step)
    log(f"[{name}] {len(orients)} orientations x {len(reps)} site kinds ({', '.join((s['kind'] for s in reps))})")
    cell = slab.cell.array
    tol = settings['dedupe_rmsd_A']
    candidates, kept = ([], [])
    order = sorted(set(symbols))
    for site in reps:
        n_site = 0
        for o in orients:
            a, contact = _place(
                slab,
                symbols,
                o['positions'],
                site['position'],
                structure,
                settings['contact_scale']
            )
            ads_pos = a.positions[n:]
            key = np.concatenate([np.sort(ads_pos[
                [i for i, s in enumerate(symbols) if s == el],
                2
            ]) for el in order])
            if any((
                np.max(np.abs(key - k)) < 2 * tol and equivalent(p, ads_pos, symbols, ops, cell, tol) for k,
                p in kept
            )):
                continue
            kept.append((key, ads_pos.copy()))
            candidates.append((site, a, contact, o))
            n_site += 1
        log(f"[{name}] site {site['kind']}: {n_site} distinct starts")
    log(f'[{name}] surface symmetry operations: {len(ops)}; distinct starts total {len(candidates)}')
    if limit:
        candidates = candidates[:limit]
    ladder = settings['height_ladder'] if contact_scale is None else None
    factory = ModelFactory(lambda: None)
    calc = FAIRChemCalculator(factory.predictor, task_name=settings['task'])
    rows = []
    fh = (out_dir / 'candidates.jsonl').open('w')
    top_z = float(slab.positions[:, 2].max())

    def relax(start_id, site, o, scale):
        a, contact = _place(
            slab,
            symbols,
            o['positions'],
            site['position'],
            structure,
            scale
        )
        a.set_constraint(FixAtoms(indices=fixed_now))
        a.calc = calc
        start = a.positions[n:].copy()
        t0 = time.monotonic()
        c0 = factory.calls
        opt = BFGS(a, maxstep=settings['maxstep_A'], logfile=None)
        try:
            ok = bool(opt.run(fmax=settings['fmax_eV_A'], steps=settings['max_steps']))
            err = None
        except Exception as exc:
            ok, err = (False, f'{type(exc).__name__}: {exc}')
        energy = float(a.get_potential_energy()) if err is None else None
        fmax = float(np.linalg.norm(a.get_forces()[n:], axis=1).max()) if err is None else None
        labels = label(
            a,
            list(range(n)),
            list(range(n, len(a))),
            reference_edges=ref_edges,
            reference_bond_lengths=ref_lengths,
            fixed_slab_indices=fixed_normal,
            clean_slab_positions=clean,
            metal_frozen=not settings['metal']['free_layers'],
            sites=all_sites
        )
        row = {
            'index': len(rows),
            'start_id': start_id,
            'species': name,
            'site_kind': site['kind'],
            'site_id': site['site_id'],
            'site_xy': site['position'][:2],
            'feature': o['feature'],
            'feature_atoms': o['members'],
            'azimuth_deg': o['azimuth_deg'],
            'contact_scale': scale,
            'start_lowest_atom_above_top_layer_A': round(float(start[:, 2].min() - top_z), 3),
            'converged': ok,
            'error': err,
            'steps': int(opt.nsteps),
            'model_calls': factory.calls - c0,
            'seconds': round(time.monotonic() - t0, 3),
            'energy_eV': energy,
            'final_fmax_eV_A': fmax,
            'labels': labels,
            'outcome': outcome(labels) if err is None else 'error',
            'start_adsorbate_positions': start.round(5).tolist(),
            'final_adsorbate_positions': a.positions[n:].round(5).tolist()
        }
        if settings['metal']['free_layers']:
            row['final_slab_positions'] = a.positions[:n].round(5).tolist()
        rows.append(row)
        fh.write(json.dumps(row) + '\n')
        fh.flush()
        a.calc = None
        return row['outcome']
    placeholder = binder is not None
    for start_id, (site, _, _, o) in enumerate(candidates):
        if start_id % 10 == 0:
            gpu_guard(log=log)
        if ladder is None:
            relax(start_id, site, o, settings['contact_scale'])
            continue
        done = {}
        while (scale := next_height(
            settings['mode'],
            placeholder,
            done,
            settings
        )) is not None:
            done[scale] = relax(start_id, site, o, scale)
    fh.close()
    del calc, factory
    ok_rows = [r for r in rows if r['energy_eV'] is not None and r['converged']]
    best = min(ok_rows, key=lambda r: r['energy_eV']) if ok_rows else None
    by_mode, by_outcome = ({}, {})
    for r in ok_rows:
        m = r['labels']['binding']['mode']
        if m not in by_mode or r['energy_eV'] < by_mode[m]['energy_eV']:
            by_mode[m] = r
        c = r['outcome']
        if c not in by_outcome or r['energy_eV'] < by_outcome[c]['energy_eV']:
            by_outcome[c] = r
    per_start = {}
    for r in rows:
        per_start[r['start_id']] = per_start.get(r['start_id'], 0) + 1
    summary = {
        'species': name,
        'surface': slab_record['name'],
        'search_settings': settings,
        'site_kinds': [s['kind'] for s in reps],
        'site_summary': {k: v for k, v in site_summary.items() if k != 'normal'},
        'n_orientations': len(orients),
        'n_starts': len(candidates),
        'n_candidates': len(rows),
        'relaxations_per_start': {str(k): sum((v == k for v in per_start.values())) for k in range(1, 6)},
        'n_converged': sum((r['converged'] for r in rows)),
        'n_errors': sum((r['error'] is not None for r in rows)),
        'wall_seconds': round(time.monotonic() - t_start, 1),
        'outcome_counts': {c: sum((r['outcome'] == c for r in rows)) for c in OUTCOMES},
        'label_counts': {k: sum((bool(r['labels'][k]['flag']) for r in rows)) for k in (
            'dissociation',
            'desorption',
            'intercalation'
        )},
        'binding_mode_counts': {m: sum((r['labels']['binding']['mode'] == m for r in rows)) for m in ('bonded', 'weak_contact', 'detached')},
        'best_by_outcome': {c: _brief(r, rows) for c, r in by_outcome.items()},
        'best': _brief(best, rows),
        'best_by_binding_mode': {m: _brief(r, rows) for m, r in by_mode.items()},
        'best_intact': _brief(
            min(
                (r for r in ok_rows if not r['labels']['dissociation']['flag']),
                key=lambda r: r['energy_eV'],
                default=None
            ),
            rows
        ),
        'gas': {k: mol_rec[k] for k in ('charge', 'spin', 'spin_reason')}
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    if best is not None:
        _write_best(slab, symbols, best, out_dir / 'best.extxyz')
    log(f"[{name}] done {len(candidates)} starts / {len(rows)} relaxations in {summary['wall_seconds']} s; outcomes {summary['outcome_counts']}")
    return summary


OUTCOMES = ('bound', 'floating', 'dissociated', 'error')


def next_height(mode, placeholder, done, settings=SEARCH):
    if mode == 'general':
        lad = settings['height_ladder']
        hi, lo = lad['ends']
        mid = lad['middle']
        for s in (hi, lo):
            if s not in done:
                return s
        if mid not in done:
            return mid
        for (u, v), refine in zip(((hi, mid), (mid, lo)), lad['refine']):
            if done[u] != done[v] and refine not in done:
                return refine
        return None
    if mode == 'fast':
        fast = settings['fast_heights']
        raise_if = fast['raise_if']['placeholder (Cl) molecule' if placeholder else 'no placeholder']
        seq = fast['sequence']
        for i, s in enumerate(seq):
            if s in done:
                continue
            if i == 0 or done.get(seq[i - 1]) in raise_if:
                return s
            return None
        return None
    raise ValueError(f'unknown mode {mode!r}')


def outcome(labels):
    if labels['dissociation']['flag']:
        return 'dissociated'
    if labels['desorption']['flag']:
        return 'floating'
    return 'bound'


def _brief(r, rows):
    if r is None:
        return None
    lab = r['labels']
    return {
        'index': r['index'],
        'energy_eV': r['energy_eV'],
        'delta_to_min_eV': round(
            r['energy_eV'] - min((x['energy_eV'] for x in rows if x['energy_eV'] is not None)),
            4
        ),
        'start_site': r['site_kind'],
        'feature': r['feature'],
        'feature_atoms': r['feature_atoms'],
        'azimuth_deg': r['azimuth_deg'],
        'converged': r['converged'],
        'binding_mode': lab['binding']['mode'],
        'bonded_elements': lab['binding']['bonded_elements'],
        'contacts': lab['binding']['contacts'],
        'flags': {k: lab[k]['flag'] for k in (
            'dissociation',
            'desorption',
            'intercalation'
        )}
    }


def _write_best(slab, symbols, row, path):
    from ase import Atoms
    from ase.io import write
    a = slab.copy()
    a += Atoms(symbols, positions=row['final_adsorbate_positions'])
    a.info.update(
        energy_eV=row['energy_eV'],
        site_start=row['site_kind'],
        feature=row['feature']
    )
    write(str(path), a)
