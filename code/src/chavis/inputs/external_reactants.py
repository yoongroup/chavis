"""Same-pool comparison: use an external set of relaxed reactant records (the enumeration
baseline's 30 random placements of one reaction and seed) in place of the CHAVIS adsorption search (wf2).

Everything after wf2 is unchanged CHAVIS: own structure labels, symmetry-merged minima, wf3 refinement
(fmax 0.01, bottom two layers fixed), selection per binding mode, product construction, pairing, TS search
and validation. The records are converted into the wf2 candidate format for one species; the relaxed metal
coordinates of the records are kept (final_slab_positions). Only the gas-phase molecule relaxation touches
the model here; the imported reactant preparation cost is charged separately by the campaign.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np

SCHEMA = 'chavis_external_reactants_v1'
SOURCE_KIND = 'enumeration_baseline_reactant_records'
CALLS_DEFINITION = ('baseline reactant placements and relaxations (endpoint_phase_costs.json reactant.calls), '
                    'charged once per reaction and seed; the baseline product and frame preparation is not imported')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve_records(seed_dir, rid):
    """Completed endpoints artifact root of one reaction in a baseline seed run; every pinned file is re-hashed."""
    checkpoint = Path(seed_dir) / 'reactions' / rid / 'checkpoint.json'
    doc = read_json(checkpoint)
    entry = (doc.get('completed_stages') or {}).get('endpoints')
    if not entry or entry.get('result', {}).get('status') != 'complete':
        raise ValueError(f'baseline endpoints stage is not complete for {rid}: {checkpoint}')
    root = Path(entry['result']['artifact_root'])
    if not root.is_dir():
        raise FileNotFoundError(f'baseline endpoint artifacts missing: {root}')
    for rel, digest in entry['artifact_hashes'].items():
        if sha256(root / rel) != digest:
            raise ValueError(f'baseline endpoint artifact changed: {root / rel}')
    return root, entry, sha256(checkpoint)


def slab_permutation(reference_positions, cell, positions, tol=1e-3):
    """perm[i] = index of the external slab atom sitting at CHAVIS slab position i (minimum image in the cell)."""
    ref = np.asarray(reference_positions, float)
    ext = np.asarray(positions, float)
    cell = np.asarray(cell, float)
    if ref.shape != ext.shape:
        raise ValueError(f'slab atom count differs: {len(ref)} vs {len(ext)}')
    inv = np.linalg.inv(cell)
    d = (ref @ inv)[:, None, :] - (ext @ inv)[None, :, :]
    d -= np.round(d)
    dist = np.linalg.norm(d @ cell, axis=2)
    perm = dist.argmin(axis=1)
    if len(set(perm.tolist())) != len(ref) or float(dist.min(axis=1).max()) > tol:
        raise ValueError('slab atoms do not match the CHAVIS slab record (cell, order or positions)')
    return [int(i) for i in perm]


def adsorbate_permutation(symbols, reference_edges, ext_symbols, ext_edges):
    """perm[i] = external index of CHAVIS adsorbate atom i; elements and the bond graph must agree.

    Edges are index pairs in their own local index space. Every permutation within each element is tried
    (small molecules), so hydrogens bound to different heavy atoms are told apart by the bond graph."""
    symbols, ext_symbols = list(symbols), list(ext_symbols)
    if sorted(symbols) != sorted(ext_symbols):
        raise ValueError(f'adsorbate composition differs: {symbols} vs {ext_symbols}')
    want = {tuple(sorted(map(int, e))) for e in reference_edges}
    have = {tuple(sorted(map(int, e))) for e in ext_edges}
    slots, groups = {}, {}
    for i, s in enumerate(symbols):
        slots.setdefault(s, []).append(i)
    for j, s in enumerate(ext_symbols):
        groups.setdefault(s, []).append(j)
    elements = sorted(groups)
    for combo in itertools.product(*(itertools.permutations(groups[e]) for e in elements)):
        perm = [None] * len(symbols)
        for e, assignment in zip(elements, combo):
            for i, j in zip(slots[e], assignment):
                perm[i] = j
        if {tuple(sorted((perm[a], perm[b]))) for a, b in want} == have:
            return [int(j) for j in perm]
    raise ValueError('no element- and bond-graph-consistent atom mapping between the record and the CHAVIS molecule')


def _energy_of(atoms, fallback):
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return fallback


def import_records(name, recipe, record, out_dir, source_root, *, reaction, seed, log=print):
    """Write <out_dir>/candidates.jsonl, summary.json, best.extxyz, external_import.json for species `name`."""
    from ase import Atoms
    from ase.io import read
    from chavis.inputs.adsorbate_search import (SEARCH, _brief, _site_representatives, _write_best,
                                                gas_molecule, outcome)
    from chavis.judgment.structure_labels import adsorbate_graph, label, reference_lengths
    from chavis.surface.metal_mode import describe
    from chavis.surface.slab_store import to_atoms
    out_dir, source_root = Path(out_dir), Path(source_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    slab = to_atoms(record)
    n = len(slab)
    structure = record['structure']
    clean = slab.positions.copy()
    cell = slab.cell.array
    fixed_normal = list(structure['fixed_indices'])
    top_z = float(clean[:, 2].max())
    symbols, mol, binder, mol_rec = gas_molecule(name, recipe, out_dir / 'gas', log)
    gas_atoms = Atoms(symbols, positions=mol)
    gas_edges = adsorbate_graph(gas_atoms, list(range(len(symbols))))
    ref_edges = {(a + n, b + n) for a, b in gas_edges}
    ref_lengths = {(a + n, b + n): v for (a, b), v in reference_lengths(gas_atoms, gas_edges).items()}
    reps, all_sites, site_summary = _site_representatives(structure)
    costs = read_json(source_root / 'endpoint_phase_costs.json')
    meta = sorted((read_json(p) for p in (source_root / 'endpoints').glob('reactant_[0-9][0-9].json')),
                  key=lambda r: r['index'])
    if not meta:
        raise FileNotFoundError(f'no reactant records under {source_root}')
    metal = describe(record, '3,4')
    metal['note'] = 'coordinates as relaxed by the baseline endpoint stage (bottom 18 atoms fixed); refined again in wf3'
    log(f'[{name}] importing {len(meta)} baseline reactant records of {reaction} seed {seed} from {source_root}')
    rows, audit = [], []
    with (out_dir / 'candidates.jsonl').open('w') as fh:
        for r in meta:
            k = int(r['index'])
            stem = source_root / 'endpoints' / f'reactant_{k:02d}'
            files = {s: stem.with_name(f'{stem.name}_{s}.extxyz') for s in ('initial', 'final')}
            entry = {'baseline_index': k, 'mapping': None,
                     'files': {s: {'path': str(p), 'sha256': sha256(p) if p.is_file() else None} for s, p in files.items()},
                     'baseline': {kk: r.get(kk) for kk in ('converged', 'steps', 'energy_eV', 'max_free_force_eV_A',
                                                            'anomaly', 'calls', 'seconds', 'error', 'failure_class')}}
            row = {'index': len(rows), 'start_id': k, 'species': name, 'site_kind': 'external', 'site_id': f'external_{k:02d}',
                   'site_xy': None, 'feature': 'external_record', 'feature_atoms': [], 'azimuth_deg': None, 'contact_scale': None,
                   'start_lowest_atom_above_top_layer_A': None, 'converged': bool(r.get('converged')), 'error': r.get('error'),
                   'steps': r.get('steps'), 'model_calls': int(r.get('calls') or 0), 'seconds': r.get('seconds'),
                   'energy_eV': None, 'final_fmax_eV_A': r.get('max_free_force_eV_A'), 'labels': None, 'outcome': 'error',
                   'start_adsorbate_positions': None, 'final_adsorbate_positions': None, 'final_slab_positions': None,
                   'external': entry}
            try:
                if r.get('error') or not files['final'].is_file() or not files['initial'].is_file():
                    raise ValueError(r.get('error') or 'record has no initial/final structure')
                init, fin = read(str(files['initial'])), read(str(files['final']))
                for a in (init, fin):
                    if not np.allclose(a.cell.array, cell, atol=1e-6):
                        raise ValueError('cell differs from the CHAVIS slab record')
                if init.get_chemical_symbols() != fin.get_chemical_symbols():
                    raise ValueError('initial and final atom orders differ')
                tags = init.get_tags()
                ext_slab = [i for i, t in enumerate(tags) if t != 2]
                ext_ads = [i for i, t in enumerate(tags) if t == 2]
                if len(ext_slab) != n or not ext_ads:
                    raise ValueError(f'expected {n} slab atoms and a tagged adsorbate, found {len(ext_slab)} and {len(ext_ads)}')
                sperm = slab_permutation(clean, cell, init.positions[ext_slab])
                ext_sym = [init.get_chemical_symbols()[i] for i in ext_ads]
                local = {i: j for j, i in enumerate(ext_ads)}
                ext_edges = [(local[a], local[b]) for a, b in adsorbate_graph(init, ext_ads)]
                aperm = adsorbate_permutation(symbols, gas_edges, ext_sym, ext_edges)
                order = [ext_slab[j] for j in sperm] + [ext_ads[j] for j in aperm]
                start, final = init.positions[order], fin.positions[order]
                atoms = slab.copy()
                atoms += Atoms(symbols, positions=final[n:])
                atoms.positions[:n] = final[:n]
                energy = float(r['energy_eV']) if r.get('energy_eV') is not None else _energy_of(fin, None)
                if energy is None:
                    raise ValueError('record has no energy')
                labels = label(atoms, list(range(n)), list(range(n, len(atoms))), reference_edges=ref_edges,
                               reference_bond_lengths=ref_lengths, fixed_slab_indices=fixed_normal,
                               clean_slab_positions=clean, metal_frozen=False, sites=all_sites)
                lead = binder if binder is not None else 0
                row.update(energy_eV=energy, labels=labels, outcome=outcome(labels),
                           site_xy=[float(x) for x in start[n + lead][:2]],
                           start_lowest_atom_above_top_layer_A=round(float(start[n:, 2].min() - top_z), 3),
                           start_adsorbate_positions=start[n:].round(5).tolist(),
                           final_adsorbate_positions=final[n:].round(5).tolist(),
                           final_slab_positions=final[:n].round(5).tolist())
                entry.update(mapping='ok', slab_permutation=sperm, adsorbate_permutation=aperm,
                             energy_file_eV=_energy_of(fin, None),
                             metal_max_displacement_A=round(float(np.linalg.norm(final[:n] - clean, axis=1).max()), 4))
            except Exception as exc:
                # Recorded for checking; the campaign's check_import refuses any pool with a failed record.
                row.update(error=f'{type(exc).__name__}: {exc}', converged=False, outcome='error')
                entry.update(mapping='failed', mapping_error=row['error'])
                log(f'[{name}] record {k:02d} could not be converted: {row["error"]}')
            rows.append(row)
            audit.append(entry)
            fh.write(json.dumps(row) + '\n')
    ok_rows = [r for r in rows if r['energy_eV'] is not None and r['converged']]
    best = min(ok_rows, key=lambda r: r['energy_eV']) if ok_rows else None
    by_mode, by_outcome = {}, {}
    for r in ok_rows:
        m = r['labels']['binding']['mode']
        if m not in by_mode or r['energy_eV'] < by_mode[m]['energy_eV']:
            by_mode[m] = r
        c = r['outcome']
        if c not in by_outcome or r['energy_eV'] < by_outcome[c]['energy_eV']:
            by_outcome[c] = r
    labelled = [r for r in rows if r['labels'] is not None]
    settings = {**SEARCH, 'mode': 'external_records', 'source': SOURCE_KIND, 'source_root': str(source_root),
                'metal': metal, 'azimuth_step_used_deg': None,
                'placement': 'baseline random_site_heuristic_placement records; no CHAVIS orientation enumeration'}
    imported = {'reactant_calls': int(costs['reactant']['calls']), 'reactant_seconds': float(costs['reactant']['seconds']),
                'baseline_placements': costs['reactant'].get('placements'), 'n_records': len(meta),
                'n_mapped': sum(e['mapping'] == 'ok' for e in audit), 'n_mapping_failed': sum(e['mapping'] == 'failed' for e in audit)}
    summary = {
        'species': name, 'surface': record['name'], 'search_settings': settings,
        'site_kinds': [s['kind'] for s in reps], 'site_summary': {k: v for k, v in site_summary.items() if k != 'normal'},
        'n_orientations': None, 'n_starts': len(meta), 'n_candidates': len(rows),
        'relaxations_per_start': {'1': len(rows)}, 'n_converged': sum(r['converged'] for r in rows),
        'n_errors': sum(r['error'] is not None for r in rows), 'wall_seconds': round(time.monotonic() - t_start, 1),
        'outcome_counts': {c: sum(r['outcome'] == c for r in rows) for c in ('bound', 'floating', 'dissociated', 'error')},
        'label_counts': {k: sum(bool(r['labels'][k]['flag']) for r in labelled) for k in ('dissociation', 'desorption', 'intercalation')},
        'binding_mode_counts': {m: sum(r['labels']['binding']['mode'] == m for r in labelled) for m in ('bonded', 'weak_contact', 'detached')},
        'best_by_outcome': {c: _brief(r, rows) for c, r in by_outcome.items()}, 'best': _brief(best, rows),
        'best_by_binding_mode': {m: _brief(r, rows) for m, r in by_mode.items()},
        'best_intact': _brief(min((r for r in ok_rows if not r['labels']['dissociation']['flag']), key=lambda r: r['energy_eV'], default=None), rows),
        'gas': {k: mol_rec[k] for k in ('charge', 'spin', 'spin_reason')}, 'imported': imported}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    if best is not None:
        _write_best(slab, symbols, best, out_dir / 'best.extxyz')
    manifest = {'schema': SCHEMA, 'species': name, 'reaction': reaction, 'seed': seed, 'source_kind': SOURCE_KIND,
                'source_dir': str(source_root), 'calls_definition': CALLS_DEFINITION,
                'imported_reactant_calls': imported['reactant_calls'], 'imported_reactant_seconds': imported['reactant_seconds'],
                'baseline_placements': imported['baseline_placements'], 'n_records': imported['n_records'],
                'n_mapped': imported['n_mapped'], 'n_mapping_failed': imported['n_mapping_failed'],
                'slab_record': {'name': record['name'], 'key_sha256': record.get('key_sha256')},
                'phase_costs_sha256': sha256(source_root / 'endpoint_phase_costs.json'), 'records': audit}
    (out_dir / 'external_import.json').write_text(json.dumps(manifest, indent=1, default=str))
    log(f"[{name}] imported {imported['n_mapped']}/{imported['n_records']} records; outcomes {summary['outcome_counts']}; "
        f"baseline reactant preparation {imported['reactant_calls']} calls charged separately")
    return summary
