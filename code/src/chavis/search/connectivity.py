from __future__ import annotations
from .geometry import RADII, cross, norm, parse_indices, sub, validate
from chavis.judgment.policy import V2_PROFILE, bond_intact


def displacement(a, b, structure):
    vector = sub(b, a)
    if norm(cross(structure['cell'][0], structure['cell'][1])) > 1e-10:
        from .surface_sites import minimum_vector
        return minimum_vector(vector, structure)
    return vector


def check_pair(pair, graph):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError('Bond needs two atom indices.')
    i, j = map(int, pair)
    if i == j or i not in graph or j not in graph:
        raise ValueError('Choose two different adsorbate atoms, not slab atoms.')
    return (i, j)


def connectivity(
    structure,
    ads,
    scale=1.2,
    backend='auto',
    overrides=None,
    *,
    criterion_profile=V2_PROFILE
):
    structure = validate(structure)
    ads = parse_indices(ads, len(structure['symbols']))
    if not 0.9 <= float(scale) <= 1.6:
        raise ValueError('Bond cutoff multiplier must be 0.9..1.6.')
    graph = {i: set() for i in ads}
    name = 'covalent-radius fallback (legacy)'
    if backend in ('auto', 'ase'):
        try:
            from ase import Atoms
            from ase.neighborlist import neighbor_list, natural_cutoffs
            atoms = Atoms(
                [structure['symbols'][i] for i in ads],
                positions=[structure['positions'][i] for i in ads],
                cell=structure['cell'],
                pbc=[structure['pbc'][0], structure['pbc'][1], False]
            )
            ii, jj, dd = neighbor_list(
                'ijd',
                atoms,
                natural_cutoffs(atoms, mult=float(scale)),
                self_interaction=False
            )
            for i, j, distance in zip(ii, jj, dd):
                if i != j and distance > 0.25:
                    graph[ads[int(i)]].add(ads[int(j)])
            name = 'ASE neighbor_list / natural_cutoffs'
        except ImportError as exc:
            if backend == 'ase':
                raise ValueError('ASE is not installed. Run INSTALL_SCIENCE.command.') from exc
    if name.startswith('covalent'):
        from itertools import combinations
        for i, j in combinations(ads, 2):
            distance = norm(displacement(
                structure['positions'][i],
                structure['positions'][j],
                structure
            ))
            if 0.25 < distance and (bond_intact(
                distance,
                'H',
                'H',
                criterion_profile=criterion_profile
            ) if structure['symbols'][i] == structure['symbols'][j] == 'H' and criterion_profile == V2_PROFILE else distance < float(scale) * (RADII.get(structure['symbols'][i], 0.8) + RADII.get(structure['symbols'][j], 0.8))):
                graph[i].add(j)
                graph[j].add(i)
    if criterion_profile == V2_PROFILE:
        from itertools import combinations
        for i, j in combinations(ads, 2):
            if structure['symbols'][i] == structure['symbols'][j] == 'H':
                distance = norm(displacement(
                    structure['positions'][i],
                    structure['positions'][j],
                    structure
                ))
                if bond_intact(distance, 'H', 'H', criterion_profile=criterion_profile):
                    graph[i].add(j)
                    graph[j].add(i)
                else:
                    graph[i].discard(j)
                    graph[j].discard(i)
    overrides = overrides or {}
    for key in ('remove', 'add'):
        for pair in overrides.get(key, []):
            i, j = check_pair(pair, graph)
            if key == 'remove':
                graph[i].discard(j)
                graph[j].discard(i)
            else:
                graph[i].add(j)
                graph[j].add(i)
    return (graph, name)
