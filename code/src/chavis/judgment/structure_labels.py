from __future__ import annotations
import numpy as np
from ase.data import covalent_radii, atomic_numbers
DESORPTION_SCALE = 1.8
INTERCALATION_SCALE = 1.0
DEFORMATION_A = 1.0
SUBSURFACE_TOL_A = 0.3
TOP_LAYER_BAND_A = 0.5
BONDED_SCALE = 1.25
SITE_MATCH_A = 0.6


def _r(sym):
    return float(covalent_radii[atomic_numbers[sym]])


def _mic_vectors(atoms, i, js):
    from ase.geometry import find_mic
    d = atoms.positions[js] - atoms.positions[i]
    v, dist = find_mic(d, atoms.cell, pbc=atoms.pbc)
    return (v, dist)


def adsorbate_graph(atoms, ads, reference_lengths=None):
    from chavis.judgment.policy import bond_threshold, V3_PROFILE
    sym = atoms.get_chemical_symbols()
    reference_lengths = reference_lengths or {}
    edges = set()
    for a in ads:
        others = [b for b in ads if b > a]
        if not others:
            continue
        _, dist = _mic_vectors(atoms, a, others)
        for b, d in zip(others, dist):
            thr = bond_threshold(
                sym[a],
                sym[b],
                reference_length_A=reference_lengths.get((a, b)),
                criterion_profile=V3_PROFILE
            )
            if d < thr:
                edges.add((a, b))
    return edges


def reference_lengths(atoms, edges):
    return {tuple(sorted(e)): float(atoms.get_distance(*e, mic=bool(atoms.pbc.any()))) for e in edges}


def _components(ads, edges):
    parent = {a: a for a in ads}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for a in ads:
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values())


def label(
    atoms,
    slab_indices,
    ads,
    *,
    reference_edges,
    fixed_slab_indices,
    reference_bond_lengths=None,
    clean_slab_positions=None,
    metal_frozen=False,
    sites=None
):
    sym = atoms.get_chemical_symbols()
    slab_indices, ads = (list(slab_indices), list(ads))
    edges = adsorbate_graph(atoms, ads, reference_bond_lengths)
    ref = {tuple(sorted(e)) for e in reference_edges}
    comps = _components(ads, edges)
    out = {'thresholds': {
        'graph_rule': 'chavis.judgment.policy.bond_threshold v3',
        'desorption_scale': DESORPTION_SCALE,
        'intercalation_scale': INTERCALATION_SCALE,
        'deformation_A': DEFORMATION_A,
        'bonded_scale': BONDED_SCALE,
        'site_match_A': SITE_MATCH_A,
        'radii': 'ASE covalent_radii'
    }}
    out['dissociation'] = {
        'flag': edges != ref,
        'fragments': len(comps),
        'bonds_lost': sorted((list(e) for e in ref - edges)),
        'bonds_formed': sorted((list(e) for e in edges - ref))
    }
    out['dissociation']['rearranged_rxn'] = bool(len(comps) == 1 and out['dissociation']['bonds_lost'] and out['dissociation']['bonds_formed'])
    nearest = {}
    for a in ads:
        v, dist = _mic_vectors(atoms, a, slab_indices)
        ratio = dist / np.array([_r(sym[a]) + _r(sym[j]) for j in slab_indices])
        k = int(np.argmin(ratio))
        nearest[a] = (
            slab_indices[k],
            float(dist[k]),
            float(ratio[k]),
            [slab_indices[j] for j in np.where(ratio < BONDED_SCALE)[0]]
        )
    detached = [c for c in comps if min((nearest[a][2] for a in c)) >= DESORPTION_SCALE]
    out['desorption'] = {
        'flag': bool(detached),
        'detached_fragments': detached,
        'min_ratio': round(min((n[2] for n in nearest.values())), 3)
    }
    inter = []
    for a in ads:
        if not fixed_slab_indices:
            break
        _, dist = _mic_vectors(atoms, a, list(fixed_slab_indices))
        lim = np.array([INTERCALATION_SCALE * (_r(sym[a]) + _r(sym[j])) for j in fixed_slab_indices])
        if np.any(dist < lim):
            inter.append(a)
    top_z = max((atoms.positions[j, 2] for j in slab_indices))
    below = [a for a in ads if atoms.positions[a, 2] < top_z - 0.3]
    ref_z = np.asarray(clean_slab_positions)[:, 2] if clean_slab_positions is not None else atoms.positions[slab_indices, 2]
    z_top = float(np.mean(ref_z[ref_z >= ref_z.max() - TOP_LAYER_BAND_A]))
    depth = {a: float(z_top - atoms.positions[a, 2]) for a in ads}
    sub = [a for a in ads if atoms.positions[a, 2] < z_top - SUBSURFACE_TOL_A]
    out['intercalation'] = {
        'flag': bool(inter or sub),
        'atoms_near_fixed_layers': inter,
        'atoms_under_top_layer': sub,
        'max_depth_below_top_layer_A': round(max(depth.values()), 3),
        'subsurface_tol_A': SUBSURFACE_TOL_A,
        'atoms_below_top_layer': below
    }
    if clean_slab_positions is not None:
        disp = np.linalg.norm(
            atoms.positions[slab_indices] - np.asarray(clean_slab_positions),
            axis=1
        )
        out['slab_deformation'] = {
            'flag': bool(np.max(disp) > DEFORMATION_A),
            'max_displacement_A': round(float(np.max(disp)), 4),
            'not_applicable_metal_frozen': bool(metal_frozen)
        }
    else:
        out['slab_deformation'] = {'flag': None, 'reason': 'no clean slab reference'}
    bonded = {a: nearest[a][3] for a in ads if nearest[a][3]}
    contacts = []
    for a, metals in sorted(bonded.items()):
        row = {
            'atom': a,
            'element': sym[a],
            'n_metal': len(metals),
            'nearest_metal_A': round(nearest[a][1], 3)
        }
        if sites:
            xy = atoms.positions[a, :2]
            best = None
            for s in sites:
                d = np.asarray(s['position'])[:2] - xy
                v, dd = _mic_2d(atoms, d)
                if best is None or dd < best[1]:
                    best = (s['kind'], dd)
            row['site'] = best[0] if best and best[1] < SITE_MATCH_A else 'between_sites'
        contacts.append(row)
    if bonded:
        mode = 'bonded'
    elif not detached:
        mode = 'weak_contact'
    else:
        mode = 'detached'
    out['binding'] = {
        'mode': mode,
        'bonded_atoms': len(bonded),
        'bonded_elements': ''.join((sym[a] for a in sorted(bonded))),
        'contacts': contacts,
        'nearest': {
            str(a): {
                'metal': n[0],
                'distance_A': round(n[1], 3),
                'ratio': round(n[2], 3)
            } for a,
            n in nearest.items()
        }
    }
    return out


def _mic_2d(atoms, d):
    from ase.geometry import find_mic
    v, dist = find_mic(
        np.array([[d[0], d[1], 0.0]]),
        atoms.cell,
        pbc=[True, True, False]
    )
    return (v[0], float(dist[0]))
