from __future__ import annotations
import collections
import json
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
from chavis.inputs.minima import DEFAULT_PARAMS as _WF3_DEFAULTS
FS_DEFAULTS = {
    'refine_fix_layers': _WF3_DEFAULTS['refine_fix_layers'],
    'fs_half_angle_deg': 30.0,
    'fs_vertical_tilt_deg': 0.5,
    'fs_rmin_A': 1.8,
    'fs_rmax_A': 3.0,
    'fs_contact_scale': 1.0,
    'fs_collision_scale': 0.7,
    'fs_max_trials': 6,
    'fs_stop_at_first_valid': False,
    'fs_spontaneous_sources': 'wf3,wf2',
    'fs_spontaneous_wf2_count': 3
}
FS_RELAX = {
    'optimizer': 'ase.optimize.BFGS',
    'maxstep_A': 0.15,
    'fmax_eV_A': 0.01,
    'max_steps': 500,
    'task': 'oc20'
}
SAME_ADS_A = 0.1
SAME_SLAB_A = 0.1


def clean_params(raw):
    from chavis.surface.metal_mode import parse
    p = {**FS_DEFAULTS, **{k: v for k, v in (raw or {}).items() if k in FS_DEFAULTS}}
    parse(p['refine_fix_layers'])
    for k in (
        'fs_half_angle_deg',
        'fs_vertical_tilt_deg',
        'fs_rmin_A',
        'fs_rmax_A',
        'fs_contact_scale',
        'fs_collision_scale'
    ):
        p[k] = float(p[k])
    if p['fs_max_trials'] != 'all':
        p['fs_max_trials'] = int(p['fs_max_trials'])
    p['fs_stop_at_first_valid'] = p['fs_stop_at_first_valid'] in (True, 'true', 'True', 1)
    if not 0 <= p['fs_half_angle_deg'] <= 180:
        raise ValueError('fs_half_angle_deg must be 0..180')
    if not 0 <= p['fs_vertical_tilt_deg'] <= 90:
        raise ValueError('fs_vertical_tilt_deg must be 0..90')
    if p['fs_rmax_A'] < p['fs_rmin_A']:
        raise ValueError('fs_rmax_A must be >= fs_rmin_A')
    if p['fs_collision_scale'] > p['fs_contact_scale']:
        raise ValueError('fs_collision_scale cannot exceed fs_contact_scale')
    if p['fs_max_trials'] != 'all' and p['fs_max_trials'] < 1:
        raise ValueError('fs_max_trials must be >= 1 or all')
    if p['fs_max_trials'] == 'all' and p['fs_stop_at_first_valid']:
        raise ValueError('all-trial collection requires fs_stop_at_first_valid=false')
    src = [s.strip() for s in str(p['fs_spontaneous_sources']).split(',') if s.strip() and s.strip() != 'none']
    if set(src) - {'wf2', 'wf3'}:
        raise ValueError('fs_spontaneous_sources: wf3, wf2, both (wf3,wf2) or none')
    p['fs_spontaneous_sources'] = ','.join(src) or 'none'
    c = str(p['fs_spontaneous_wf2_count']).strip()
    if c != 'all' and (not (c.isdigit() and int(c) >= 0)):
        raise ValueError('fs_spontaneous_wf2_count must be all or a number')
    p['fs_spontaneous_wf2_count'] = c if c == 'all' else int(c)
    return p


def reaction_bond_lost(labels, pair, symbols, expected=None, ref_edges=None, ads=None):
    d = labels['dissociation']
    if len(d['bonds_lost']) != 1 or d['bonds_formed']:
        return None
    a, b = d['bonds_lost'][0]
    if sorted((symbols[a], symbols[b])) != sorted(pair):
        return None
    if expected is not None:
        from chavis.inputs.reaction_fragments import cut_gives_products
        if not cut_gives_products(symbols, ref_edges, ads, (a, b), expected):
            return None
    return [a, b]


class Species:

    def __init__(self, species_dir, record):
        from ase import Atoms
        from chavis.inputs.adsorbate_search import _site_representatives
        from chavis.judgment.structure_labels import adsorbate_graph, reference_lengths
        from chavis.surface import slab_store
        self.dir = Path(species_dir)
        self.record = record
        mol = json.loads((self.dir / 'gas' / 'molecule.json').read_text())
        self.sym = mol['symbols']
        self.slab = slab_store.to_atoms(record)
        n = self.n = len(self.slab)
        self.clean = self.slab.positions.copy()
        gas = Atoms(self.sym, positions=mol['positions_after_placeholder_removal'])
        ge = adsorbate_graph(gas, list(range(len(self.sym))))
        self.ref = {(a + n, b + n) for a, b in ge}
        self.rl = {
            (a + n, b + n): v for (a, b),
            v in reference_lengths(gas, ge).items()
        }
        _, self.sites, _ = _site_representatives(record['structure'])
        self.normal_fixed = list(record['structure']['fixed_indices'])
        self.all_symbols = self.slab.get_chemical_symbols() + list(self.sym)
        self.calc = self.factory = None

    def atoms(self, slab_positions, ads_positions):
        from ase import Atoms
        a = self.slab.copy()
        a += Atoms(self.sym, positions=ads_positions)
        a.positions[:self.n] = slab_positions
        return a

    def structure(self, slab_positions, ads_positions):
        st = deepcopy(self.record['structure'])
        n = self.n
        st['symbols'] = list(st['symbols']) + list(self.sym)
        st['positions'] = [list(map(float, x)) for x in slab_positions] + [list(map(float, x)) for x in ads_positions]
        st['atom_ids'] = list(range(len(st['symbols'])))
        st['adsorbate_indices'] = list(range(n, len(st['symbols'])))
        st['meta'] = {
            **st.get('meta', {}),
            'adsorbate_indices': st['adsorbate_indices']
        }
        st['fixed_indices'] = []
        for k in ('energy', 'layers', 'tags'):
            st.pop(k, None)
        return st

    def ensure_calc(self):
        if self.calc is None:
            from fairchem.core import FAIRChemCalculator
            from automation_search.uma_model import ModelFactory
            self.factory = ModelFactory(lambda: None)
            self.calc = FAIRChemCalculator(
                self.factory.predictor,
                task_name=FS_RELAX['task']
            )
        return self.calc

    def relax(self, atoms, fix_layers):
        from ase.constraints import FixAtoms
        from ase.optimize import BFGS
        from chavis.inputs.adsorbate_search import outcome
        from chavis.judgment.structure_labels import label
        from chavis.surface.metal_mode import fixed_indices
        self.ensure_calc()
        n = self.n
        atoms.set_constraint(FixAtoms(indices=fixed_indices(self.record, fix_layers)))
        atoms.calc = self.calc
        t0 = time.monotonic()
        c0 = self.factory.calls
        opt = BFGS(atoms, maxstep=FS_RELAX['maxstep_A'], logfile=None)
        try:
            ok = bool(opt.run(fmax=FS_RELAX['fmax_eV_A'], steps=FS_RELAX['max_steps']))
            err = None
        except Exception as exc:
            ok, err = (False, f'{type(exc).__name__}: {exc}')
        e = float(atoms.get_potential_energy()) if err is None else None
        lab = label(
            atoms,
            list(range(n)),
            list(range(n, len(atoms))),
            reference_edges=self.ref,
            reference_bond_lengths=self.rl,
            fixed_slab_indices=self.normal_fixed,
            clean_slab_positions=self.clean,
            metal_frozen=False,
            sites=self.sites
        )
        atoms.calc = None
        return {
            'converged': ok,
            'error': err,
            'steps': int(opt.nsteps),
            'model_calls': self.factory.calls - c0,
            'seconds': round(time.monotonic() - t0, 2),
            'energy_eV': e,
            'labels': lab,
            'outcome': outcome(lab) if err is None else 'error',
            'final_slab_positions': atoms.positions[:n].round(5).tolist(),
            'final_adsorbate_positions': atoms.positions[n:].round(5).tolist()
        }

    def close(self):
        self.calc = self.factory = None


def bond_choices(sp, is_row, reaction):
    from chavis.inputs.seed_geometry import cleavage
    from chavis.inputs.reaction_fragments import expected_products, cut_gives_products
    from chavis.search.geometry import to_ase
    a = to_ase(sp.structure(
        is_row['final_slab_positions'],
        is_row['final_adsorbate_positions']
    ))
    expected = expected_products(reaction)
    ads = list(range(sp.n, sp.n + len(sp.sym)))
    out, seen = ([], set())
    for k in range(64):
        try:
            anchor, moving, edges = cleavage(a, reaction, k)
        except ValueError:
            break
        if (anchor, moving[0]) in seen:
            break
        seen.add((anchor, moving[0]))
        if cut_gives_products(
            sp.all_symbols,
            edges,
            ads,
            (anchor, moving[0]),
            expected
        ):
            out.append((anchor, moving))
    return out


def generate(sp, is_row, anchor, moving, p):
    from chavis.search.surface_sites import generate_surface
    from chavis.search.collision import GeometryError, minimum_vector
    from chavis.search.geometry import basis
    st = sp.structure(
        is_row['final_slab_positions'],
        is_row['final_adsorbate_positions']
    )
    gen = dict(
        mode='cleavage',
        anchor=anchor,
        moving=moving,
        adsorbate=st['adsorbate_indices'],
        height_mode='drop',
        contact_scale=p['fs_contact_scale'],
        collision_scale=p['fs_collision_scale'],
        domain='sector',
        half_angle=p['fs_half_angle_deg'],
        rmin=p['fs_rmin_A'],
        rmax=p['fs_rmax_A']
    )
    vertical = False
    tilt = None
    threshold = p.get('fs_vertical_tilt_deg', FS_DEFAULTS['fs_vertical_tilt_deg'])
    reason = None
    try:
        vector = np.asarray(
            minimum_vector(
                np.asarray(st['positions'][moving[0]]) - st['positions'][anchor],
                st
            ),
            float
        )
        length = np.linalg.norm(vector)
        if length > 1e-09:
            normal = np.asarray(basis(st)[2], float)
            planar = vector - np.dot(vector, normal) * normal
            tilt = float(np.degrees(np.arctan2(
                np.linalg.norm(planar),
                abs(np.dot(vector, normal))
            )))
            if threshold > 0 and tilt <= threshold + 1e-12:
                gen['half_angle'] = 180.0
                vertical = True
                reason = 'near_vertical_tilt'
        rows = generate_surface(st, gen)
    except GeometryError as exc:
        if not str(exc).startswith('Bond projection on the surface is zero'):
            return (
                [],
                {
                    'error': str(exc),
                    'vertical_bond_fallback': vertical,
                    'bond_tilt_deg': tilt,
                    'fs_vertical_tilt_deg': threshold,
                    'vertical_fallback_reason': reason,
                    'n_enumerated': 0,
                    'n_accepted': 0
                }
            )
        gen['half_angle'] = 180.0
        vertical = True
        reason = 'legacy_zero_projection'
        rows = generate_surface(st, gen)
    acc = [r for r in rows if r['accepted']]
    return (
        acc,
        {
            'error': None,
            'vertical_bond_fallback': vertical,
            'bond_tilt_deg': tilt,
            'fs_vertical_tilt_deg': threshold,
            'vertical_fallback_reason': reason,
            'n_enumerated': len(rows),
            'n_accepted': len(acc)
        }
    )


def site_round_robin(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[str(row['detail'].get('site') or 'unknown')].append(row)
    kinds = sorted(groups)
    return [groups[kind][i] for i in range(max(map(len, groups.values()), default=0)) for kind in kinds if i < len(groups[kind])]


def fs_for_reaction(
    sp,
    reaction,
    is_rows,
    refined_all,
    wf2_minima,
    wf2_rows,
    p,
    log=print
):
    from chavis.inputs.adsorbate_search import gpu_guard
    pair = reaction['broken_bond_elements'].split('-')
    from chavis.inputs.reaction_fragments import expected_products
    frag = dict(
        expected=expected_products(reaction),
        ref_edges=sorted(sp.ref),
        ads=list(range(sp.n, sp.n + len(sp.sym)))
    )
    out = []
    for is_row in is_rows:
        fix = is_row['metal']['fixed_layers']
        for vi, (anchor, moving) in enumerate(bond_choices(sp, is_row, reaction)):
            acc, info = generate(sp, is_row, anchor, moving, p)
            if info['error']:
                log(f"  {reaction['id']} IS {is_row['minimum_representative']} cut {anchor}-{moving[0]}: generator {info['error']}")
            selected = site_round_robin(acc)
            if p['fs_max_trials'] != 'all':
                selected = selected[:p['fs_max_trials']]
            info.update(
                trial_selection='site_type_round_robin',
                n_available_by_site_type=dict(collections.Counter((str(c['detail'].get('site') or 'unknown') for c in acc))),
                n_selected_by_site_type=dict(collections.Counter((str(c['detail'].get('site') or 'unknown') for c in selected)))
            )
            cut_start = len(out)
            attempted = collections.Counter()
            for ci, c in enumerate(selected):
                if len(out) % 10 == 0:
                    gpu_guard(log=log)
                pos = np.asarray(c['structure']['positions'], float)
                attempted[str(c['detail'].get('site') or 'unknown')] += 1
                res = sp.relax(sp.atoms(pos[:sp.n], pos[sp.n:]), fix)
                lost = reaction_bond_lost(res['labels'], pair, sp.all_symbols, **frag)
                out.append({
                    'source': 'generated',
                    'from_is': is_row['minimum_representative'],
                    'cut': [anchor, moving[0]],
                    'moving': moving,
                    'bond_choice': vi,
                    'trial': ci,
                    'site': c['detail'].get('site'),
                    'azimuth_offset_deg': round(
                        c['detail'].get('azimuth_offset_deg', 0),
                        2
                    ),
                    'azimuth_offset_raw_deg': c['detail'].get('azimuth_offset_deg', 0),
                    'distance_A': round(c['detail'].get('distance_A', 0), 3),
                    **info,
                    'fixed_layers': fix,
                    'bond_lost': lost,
                    'product_match': bool(res['converged'] and lost and (sorted(lost) == sorted([anchor, moving[0]]))),
                    **res
                })
                if p['fs_stop_at_first_valid'] and out[-1]['product_match']:
                    break
            for row in out[cut_start:]:
                row['n_attempted_by_site_type'] = dict(attempted)
    sources = p['fs_spontaneous_sources'].split(',')
    is_fix = is_rows[0]['metal']['fixed_layers'] if is_rows else p['refine_fix_layers']
    if 'wf3' in sources:
        for r in refined_all:
            if r['outcome'] != 'dissociated' or not r.get('converged') or (not r.get('is_representative', True)):
                continue
            lost = reaction_bond_lost(r['labels'], pair, sp.all_symbols, **frag)
            if lost:
                out.append({
                    'source': 'spontaneous_wf3',
                    'from_wf2_minimum': r['minimum_representative'],
                    'fixed_layers': r['metal']['fixed_layers'],
                    'bond_lost': lost,
                    'product_match': True,
                    **{k: r[k] for k in (
                        'converged',
                        'error',
                        'steps',
                        'energy_eV',
                        'labels',
                        'outcome',
                        'final_slab_positions',
                        'final_adsorbate_positions'
                    )}
                })
    if 'wf2' in sources and is_fix is not None:
        cand = []
        for m in wf2_minima:
            if m['outcome'] != 'dissociated':
                continue
            row = wf2_rows[m['representative']]
            if reaction_bond_lost(row['labels'], pair, sp.all_symbols, **frag):
                cand.append(m)
        cand.sort(key=lambda m: m['energy_eV'])
        cap = p['fs_spontaneous_wf2_count']
        for m in cand if cap == 'all' else cand[:cap]:
            row = wf2_rows[m['representative']]
            # Imported records (same-pool comparison) carry relaxed metal coordinates; native wf2 rows do not.
            res = sp.relax(sp.atoms(row.get('final_slab_positions') or sp.clean, row['final_adsorbate_positions']), is_fix)
            lost = reaction_bond_lost(res['labels'], pair, sp.all_symbols, **frag)
            out.append({
                'source': 'spontaneous_wf2',
                'from_wf2_minimum': m['representative'],
                'wf2_form': m['form'],
                'fixed_layers': is_fix,
                'bond_lost': lost,
                'product_match': bool(res['converged'] and lost),
                **res
            })
    dedupe(out, sp)
    return out


def dedupe(rows, sp):
    from chavis.surface.symmetry import surface_operations, equivalent_with_slab
    ops = surface_operations(sp.slab)
    cell = sp.slab.cell.array
    slab_sym = sp.slab.get_chemical_symbols()
    reps = []
    order = sorted(
        range(len(rows)),
        key=lambda i: (rows[i]['energy_eV'] is None, rows[i]['energy_eV'] or 0.0)
    )
    for i in order:
        r = rows[i]
        if not r['product_match'] or r['energy_eV'] is None:
            r['fs_group'] = None
            r['fs_representative'] = False
            continue
        for g, rep in enumerate(reps):
            if abs(rep['energy_eV'] - r['energy_eV']) < 0.05 and equivalent_with_slab(
                rep['final_adsorbate_positions'],
                r['final_adsorbate_positions'],
                sp.sym,
                rep['final_slab_positions'],
                r['final_slab_positions'],
                slab_sym,
                ops,
                cell,
                SAME_ADS_A,
                SAME_SLAB_A
            ):
                r['fs_group'] = g
                r['fs_representative'] = False
                rep['group_sources'] = sorted(set(rep['group_sources']) | {r['source']})
                break
        else:
            r['fs_group'] = len(reps)
            r['fs_representative'] = True
            r['group_sources'] = [r['source']]
            reps.append(r)
    return rows
