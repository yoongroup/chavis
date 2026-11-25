from __future__ import annotations
from collections import Counter
from itertools import combinations, permutations, product
import math
import numpy as np
from .graph import POLICY_VERSION as GRAPH_POLICY_VERSION, RADII, RULES, distances as _distances, molecular_graph as _graph, reference_edges as _reference_edges, structure_context as _structure
from .policy import NUMERIC_POLICY, V2_PROFILE, V3_PROFILE, bond_intact, reaction_role_contract, surface_role
from .surface_context import selected_surface_element
POLICY_VERSION = 'ni-drm-reaction-chemical-state-test-v1'
POLICY_VERSION_V3 = 'ni-drm-reaction-chemical-state-test-v3'


def _isotopes(structure, natoms):
    value = structure.get(
        'isotopes',
        structure.get('atom_isotopes', structure.get('meta', {}).get('isotopes'))
    )
    if value is None:
        return [None] * natoms
    if not isinstance(value, list) or len(value) != natoms:
        raise ValueError('INVALID_ISOTOPE_CONTEXT')
    labels = []
    for isotope in value:
        if isotope is None:
            labels.append(None)
        elif type(isotope) is int and isotope > 0:
            labels.append(isotope)
        else:
            raise ValueError('INVALID_ISOTOPE_CONTEXT')
    return labels


def _signatures(structure):
    _, _, symbols, ids, ads, _, _ = _structure(structure)
    isotopes = _isotopes(structure, len(ids))
    return (symbols, ids, ads, [(symbols[i], isotopes[i]) for i in range(len(ids))])


def _permutations(reference, candidate):
    ref_symbols, ref_ids, ref_ads, ref_sig = _signatures(reference)
    can_symbols, can_ids, can_ads, can_sig = _signatures(candidate)
    if ref_ids != can_ids or ref_ads != can_ads or ref_symbols != can_symbols:
        raise ValueError('ATOM_ID_OR_COMPOSITION_CONTEXT_MISMATCH')
    if Counter((ref_sig[i] for i in ref_ads)) != Counter((can_sig[i] for i in can_ads)):
        raise ValueError('ISOTOPE_COMPOSITION_MISMATCH')
    if any((ref_sig[i] != can_sig[i] for i in range(len(ref_ids)))):
        raise ValueError('ATOM_ID_ISOTOPE_CONTEXT_MISMATCH')
    groups = {}
    for index in ref_ads:
        groups.setdefault(ref_sig[index], []).append(index)
    source_groups = [sorted(groups[key]) for key in sorted(
        groups,
        key=lambda x: (x[0], -1 if x[1] is None else x[1])
    )]
    maps = []
    for choices in product(*(permutations(group) for group in source_groups)):
        mapping = {
            source: destination for sources,
            destinations in zip(source_groups, choices) for source,
            destination in zip(sources, destinations)
        }
        maps.append(mapping)
    maps.sort(key=lambda mapping: (
        sum((mapping[i] != i for i in ref_ads)),
        tuple((mapping[i] for i in ref_ads))
    ))
    return maps


def _mapped_edges(structure, reference_bonds, mapping, *, criterion_profile=V2_PROFILE):
    _, _, symbols, _, ads, _, _ = _structure(structure)
    distance = _distances(structure)
    actual_edges, mapped_edges = (set(), set())
    for a, b in combinations(ads, 2):
        reference_pair = tuple(sorted((mapping[a], mapping[b])))
        if bond_intact(
            distance[a, b],
            symbols[a],
            symbols[b],
            reference_length_A=reference_bonds.get(reference_pair),
            criterion_profile=criterion_profile
        ):
            actual_edges.add(tuple(sorted((a, b))))
            mapped_edges.add(reference_pair)
    return (actual_edges, mapped_edges, distance)


def _formula(symbols):
    count = Counter(symbols)
    return ''.join((element + (str(count[element]) if count[element] > 1 else '') for element in sorted(count)))


def _fragment_states(
    structure,
    actual_edges,
    distance,
    mapping,
    *,
    criterion_profile=V2_PROFILE,
    intact_parent=False
):
    positions, cell, symbols, ids, ads, fixed, _ = _structure(structure)
    normal = np.cross(cell[0], cell[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 0:
        raise ValueError('INVALID_SLAB_NORMAL')
    normal /= norm
    if normal[2] < 0:
        normal = -normal
    slab = [i for i in range(len(positions)) if i not in ads]
    if len(slab) == 36 and len(fixed) == 18 and all((symbols[i] == 'Ni' for i in slab)):
        metal = 'Ni'
    else:
        metal = selected_surface_element(
            structure,
            positions=positions,
            cell=cell,
            symbols=symbols,
            adsorbate_indices=ads,
            fixed_indices=fixed,
            pbc=structure.get('pbc')
        )
    heights = positions @ normal
    plane = float(np.mean(heights[sorted(slab, key=lambda i: heights[i])[-9:]]))
    adjacency = {i: set() for i in ads}
    for a, b in actual_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(ads)
    fragments = []
    while remaining:
        stack = [min(remaining)]
        members = set(stack)
        while stack:
            for neighbor in sorted(adjacency[stack.pop()] - members):
                members.add(neighbor)
                stack.append(neighbor)
        remaining -= members
        members = sorted(members)
        closest = min(((
            distance[i, j] / (RADII[symbols[i]] + RADII[metal]),
            i,
            j
        ) for i in members for j in slab))
        scaled, _, nearest_ni = closest
        min_height = float(min((heights[i] - plane for i in members)))
        role = surface_role(
            scaled,
            min_height,
            criterion_profile=criterion_profile,
            intact_parent=intact_parent and len(members) == len(ads)
        )
        fragment = {
            'actual_atom_ids': [ids[i] for i in members],
            'mapped_reference_atom_ids': sorted((ids[mapping[i]] for i in members)),
            'formula': _formula((symbols[i] for i in members)),
            'role': role,
            'minimum_height_A': min_height
        }
        if metal == 'Ni':
            fragment.update(
                minimum_Ni_radius_ratio=float(scaled),
                nearest_Ni_atom_id=ids[nearest_ni]
            )
        else:
            fragment.update(
                surface_element=metal,
                minimum_metal_radius_ratio=float(scaled),
                nearest_metal_atom_id=ids[nearest_ni]
            )
        fragments.append(fragment)
    fragments.sort(key=lambda row: (
        row['mapped_reference_atom_ids'],
        row['actual_atom_ids']
    ))
    return fragments


def _state(structure, refs, mapping, *, criterion_profile=V2_PROFILE):
    actual, mapped, distance = _mapped_edges(
        structure,
        refs,
        mapping,
        criterion_profile=criterion_profile
    )
    fragments = _fragment_states(
        structure,
        actual,
        distance,
        mapping,
        criterion_profile=criterion_profile,
        intact_parent=set(mapped) == set(refs)
    )
    ids = structure['atom_ids']
    return {
        'mapped_edges_reference_atom_ids': [
            list(sorted((ids[a], ids[b]))) for a,
            b in sorted(mapped)
        ],
        'actual_edges_atom_ids': [
            list(sorted((ids[a], ids[b]))) for a,
            b in sorted(actual)
        ],
        'fragments': fragments,
        'fragment_formulas': sorted((row['formula'] for row in fragments)),
        'fragment_formula_multiplicity': dict(sorted(Counter((row['formula'] for row in fragments)).items())),
        'all_fragments_surface_contact': all((row['role'] == 'surface_contact' for row in fragments)),
        'any_fragment_detached': any((row['role'] == 'detached_molecule' for row in fragments))
    }


def _edge_set(state, ids):
    by_id = {atom_id: index for index, atom_id in enumerate(ids)}
    return {
        tuple(sorted((by_id[a], by_id[b]))) for a,
        b in state['mapped_edges_reference_atom_ids']
    }


def _site_channel(
    minus,
    plus,
    reference_parent=None,
    reference_products=None,
    orientation=None
):
    metal = 'Cu' if any((fragment.get('surface_element') == 'Cu' for state in (minus, plus) for fragment in state['fragments'])) else 'Ni'
    nearest_key = 'nearest_metal_atom_id' if metal == 'Cu' else 'nearest_Ni_atom_id'
    output_key = 'nearest_Cu_atom_ids' if metal == 'Cu' else 'nearest_Ni_atom_ids'
    observed = {
        key: {tuple(fragment['mapped_reference_atom_ids']): fragment[nearest_key] for fragment in state['fragments']} for key,
        state in (('minus', minus), ('plus', plus))
    }
    references = {}
    if orientation is not None and reference_parent is not None and (reference_products is not None):
        for side in ('minus', 'plus'):
            reference = reference_parent if orientation[side] == 'parent' else reference_products
            references[side] = {tuple(fragment['mapped_reference_atom_ids']): fragment[nearest_key] for fragment in reference['fragments']}
    different_nearest = any((
        observed[side].get(fragment) != nearest for side,
        by_fragment in references.items() for fragment,
        nearest in by_fragment.items()
    ))
    return {
        'status': 'different_or_unproven' if different_nearest else 'unknown',
        'reason': f'DIFFERENT_NEAREST_{metal.upper()}_FINGERPRINT_NO_SYMMETRY_PROOF' if different_nearest else 'NO_SAVED_SITE_SYMMETRY_COMPARISON_IN_CHEMICAL_CLASSIFIER',
        'geometric_equivalence_claimed': False,
        'distinct_adsorption_channels_not_collapsed': True,
        'endpoint_' + output_key: {
            key: [fragment[nearest_key] for fragment in state['fragments']] for key,
            state in (('minus', minus), ('plus', plus))
        },
        'reference_' + output_key: {
            side: list(by_fragment.values()) for side,
            by_fragment in references.items()
        }
    }


def _role_map(state):
    return {tuple(fragment['mapped_reference_atom_ids']): 'surface_contact' if fragment['role'] == 'weak_contact' else fragment['role'] for fragment in state['fragments']}


def _parent_role_compatible(candidate, reference, criterion_profile):
    actual = _role_map(candidate)
    expected = _role_map(reference)
    if criterion_profile in (V2_PROFILE, V3_PROFILE) and len(candidate['fragments']) == 1 and (candidate['fragments'][0]['role'] == 'weak_contact'):
        key = tuple(candidate['fragments'][0]['mapped_reference_atom_ids'])
        if key in expected:
            actual[key] = expected[key]
    return actual == expected


def _requested_roles_compatible(parent, products, contract):
    parent_fragments = parent['fragments']
    return len(parent_fragments) == 1 and parent_fragments[0]['role'] in contract['allowed_parent_roles'] and bool(products['fragments']) and all((row['role'] == contract['required_product_role'] for row in products['fragments']))


def _subtype(parent, products):
    parent_roles = ['surface_contact' if row['role'] == 'weak_contact' else row['role'] for row in parent['fragments']]
    product_roles = [row['role'] for row in products['fragments']]
    if parent_roles == ['detached_molecule'] and all((r == 'surface_contact' for r in product_roles)):
        return 'dissociative_adsorption'
    if parent_roles == ['surface_contact'] and all((r == 'surface_contact' for r in product_roles)):
        return 'surface_cleavage_recombination'
    return 'other_or_ambiguous_bond_transformation'


def _mapping_rank(role_match, mapping, adsorbate_indices, parent_side):
    return (
        0 if role_match else 1,
        sum((mapping[i] != i for i in adsorbate_indices)),
        tuple((mapping[i] for i in adsorbate_indices)),
        0 if parent_side == 'minus' else 1
    )


def _declared_edges(value, ids, ads):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError('DECLARED_EDGES_NOT_A_LIST')
    index_by_id = {atom_id: index for index, atom_id in enumerate(ids)}
    edges = set()
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any((type(atom_id) is not int or atom_id not in index_by_id for atom_id in pair)):
            raise ValueError('INVALID_DECLARED_EDGE_ATOM_IDS')
        a, b = (index_by_id[atom_id] for atom_id in pair)
        if a == b or a not in ads or b not in ads:
            raise ValueError('DECLARED_EDGE_NOT_TWO_ADSORBATE_ATOMS')
        edges.add(tuple(sorted((a, b))))
    if len(edges) != len(value):
        raise ValueError('DUPLICATE_DECLARED_EDGE')
    return edges


def _edge_ids(edges, ids):
    return [list(sorted((ids[a], ids[b]))) for a, b in sorted(edges)]


def validate_input_reaction(
    IS,
    FS,
    *,
    species,
    descriptor,
    criterion_profile=V2_PROFILE
):
    out = {
        'status': 'unknown',
        'reason': 'INPUT_REACTION_NOT_VALIDATED',
        'declared': {},
        'observed': {},
        'reference_graph': None
    }
    try:
        if not isinstance(descriptor, dict):
            return out
        _, cell, symbols, ids, ads, fixed, pbc = _structure(IS)
        _, final_cell, final_symbols, final_ids, final_ads, final_fixed, final_pbc = _structure(FS)
        if symbols != final_symbols or ids != final_ids or ads != final_ads or (fixed != final_fixed) or (pbc != final_pbc) or (not np.allclose(cell, final_cell, rtol=0, atol=1e-08)):
            raise ValueError('SAVED_IS_FS_ATOM_CONTEXT_MISMATCH')
        _permutations(IS, FS)
        pair_ids = descriptor.get('target_pair_atom_ids')
        if not isinstance(pair_ids, (list, tuple)) or len(pair_ids) != 2 or any((type(atom_id) is not int or atom_id not in ids for atom_id in pair_ids)):
            raise ValueError('DECLARED_TARGET_PAIR_MISSING_OR_INVALID')
        index_by_id = {atom_id: index for index, atom_id in enumerate(ids)}
        pair = tuple(sorted((index_by_id[atom_id] for atom_id in pair_ids)))
        if pair[0] == pair[1] or any((i not in ads for i in pair)):
            raise ValueError('DECLARED_TARGET_PAIR_NOT_TWO_ADSORBATE_ATOMS')
        declared_elements = descriptor.get('target_bond_elements')
        if declared_elements is not None:
            parts = declared_elements.split('-') if isinstance(declared_elements, str) else declared_elements
            if not isinstance(parts, (list, tuple)) or len(parts) != 2 or sorted(parts) != sorted((symbols[i] for i in pair)):
                raise ValueError('DECLARED_TARGET_BOND_ELEMENTS_DIFFER_FROM_ATOM_IDS')
        declared_is = _declared_edges(descriptor.get('expected_IS_edges'), ids, ads)
        declared_fs = _declared_edges(descriptor.get('expected_FS_edges'), ids, ads)
        direction = descriptor.get('computational_direction', 'dissociation')
        if direction not in ('dissociation', 'association'):
            return {
                **out,
                'reason': 'UNSUPPORTED_COMPUTATIONAL_DIRECTION',
                'declared': {'computational_direction': direction}
            }
        inferred_is = set(_graph(IS, {}, criterion_profile=criterion_profile)[0])
        measured_is = _distances(IS)
        if direction == 'dissociation':
            proposed = set(declared_is) if declared_is is not None else inferred_is | {pair}
            for a, b in proposed - inferred_is:
                if not bond_intact(
                    measured_is[a, b],
                    symbols[a],
                    symbols[b],
                    nonreference_scale=NUMERIC_POLICY['reference_input_radius_scale'],
                    criterion_profile=criterion_profile
                ):
                    raise ValueError('DECLARED_IS_BOND_NOT_SUPPORTED_BY_SAVED_GEOMETRY')
            observed_is = inferred_is | proposed - inferred_is
        else:
            observed_is = inferred_is
        refs = {edge: float(measured_is[edge]) for edge in observed_is}
        observed_fs = set(_graph(FS, refs, criterion_profile=criterion_profile)[0])
        out['declared'] = {
            'target_pair_atom_ids': list(pair_ids),
            'target_bond_elements': declared_elements,
            'expected_IS_edges_atom_ids': None if declared_is is None else _edge_ids(declared_is, ids),
            'expected_FS_edges_atom_ids': None if declared_fs is None else _edge_ids(declared_fs, ids),
            'requested_direction': descriptor.get('requested_direction'),
            'computational_direction': direction,
            'requested_bond_operation': 'formation' if descriptor.get('requested_direction') == 'association' else 'cleavage' if descriptor.get('requested_direction') == 'dissociation' else None,
            'computational_bond_operation': 'formation' if direction == 'association' else 'cleavage'
        }
        out['observed'] = {
            'IS_edges_atom_ids': _edge_ids(observed_is, ids),
            'FS_edges_atom_ids': _edge_ids(observed_fs, ids),
            'IS_geometry_inferred_edges_atom_ids': _edge_ids(inferred_is, ids),
            'target_pair_distance_IS_A': float(measured_is[pair]),
            'target_pair_distance_FS_A': float(_distances(FS)[pair])
        }
        out['reference_graph'] = {
            'IS_edges_atom_ids': out['observed']['IS_edges_atom_ids'],
            'FS_edges_atom_ids': out['observed']['FS_edges_atom_ids'],
            'IS_reference_bond_lengths_A': [
                {'atom_ids': list(sorted((ids[a], ids[b]))), 'r0_A': length} for (a, b),
                length in sorted(refs.items())
            ]
        }
        if declared_is is not None and declared_is != observed_is:
            raise ValueError('DECLARED_IS_EDGES_DIFFER_FROM_SAVED_GEOMETRY')
        if declared_fs is not None and declared_fs != observed_fs:
            raise ValueError('DECLARED_FS_EDGES_DIFFER_FROM_SAVED_GEOMETRY')
        expected_fs = observed_is - {pair} if direction == 'dissociation' else observed_is | {pair}
        if direction == 'dissociation' and pair not in observed_is or (direction == 'association' and pair in observed_is) or observed_fs != expected_fs:
            raise ValueError('SAVED_IS_FS_NOT_DECLARED_SINGLE_TARGET_BOND_EDIT')
        out.update(
            status='validated',
            reason='SAVED_IS_FS_MATCH_DECLARED_SINGLE_BOND_EDIT'
        )
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        out.update(status='input_reaction_mismatch', reason=str(exc))
    return out


def classify_states(
    IS,
    FS,
    minus,
    plus,
    *,
    species,
    reaction_id,
    target_pair_atom_ids,
    requested_direction='dissociation',
    input_reaction=None,
    criterion_profile=V2_PROFILE
):
    out = {
        'policy_version': POLICY_VERSION_V3 if criterion_profile == V3_PROFILE else POLICY_VERSION,
        'frozen_graph_policy_version': GRAPH_POLICY_VERSION,
        'reaction_id': reaction_id,
        'requested_direction': requested_direction,
        'status': 'unknown',
        'reason': 'NOT_CLASSIFIED',
        'orientation': None,
        'endpoint_states': {},
        'graph_evidence': {},
        'bond_edit_compatible': None,
        'bond_transformation_compatible': None,
        'reference_role_compatible': None,
        'requested_role_compatible': None,
        'role_mismatch': None,
        'reference_subtype': None,
        'observed_subtype': None,
        'channel_subtype': 'unresolved',
        'fragment_multiplicity_status': 'counted_from_frozen_molecular_graph',
        'fragment_formula_multiplicity': {'reference': None, 'observed': None},
        'electronic_fragment_spin_status': 'unavailable_not_inferred',
        'spin_resolved_chemical_identity': False,
        'site_channel': {'status': 'unknown', 'geometric_equivalence_claimed': False}
    }
    if input_reaction is not None:
        input_check = validate_input_reaction(
            IS,
            FS,
            species=species,
            descriptor=input_reaction,
            criterion_profile=criterion_profile
        )
        if input_check['status'] == 'validated' and sorted(input_reaction['target_pair_atom_ids']) != sorted(target_pair_atom_ids):
            input_check = {
                **input_check,
                'status': 'input_reaction_mismatch',
                'reason': 'CLASSIFIER_TARGET_PAIR_DIFFERS_FROM_INPUT_DESCRIPTOR'
            }
        out['input_reaction_validation'] = input_check
        out['input_reaction_mismatch'] = input_check['status'] == 'input_reaction_mismatch'
        if input_check['status'] != 'validated':
            out['reason'] = 'INPUT_REACTION_MISMATCH' if out['input_reaction_mismatch'] else 'INPUT_REACTION_UNVALIDATED'
            return out
        if input_reaction.get(
            'computational_direction',
            'dissociation'
        ) != 'dissociation':
            out['reason'] = 'COMPUTATIONAL_FORMATION_CANDIDATE_COMPARISON_UNSUPPORTED'
            return out
    try:
        if not all((isinstance(value, dict) for value in (IS, FS, minus, plus))):
            raise ValueError('MISSING_SAVED_STRUCTURE')
        if input_reaction is None:
            refs = _reference_edges(IS, species, criterion_profile=criterion_profile)
        else:
            index_by_id = {
                atom_id: index for index,
                atom_id in enumerate(IS['atom_ids'])
            }
            refs = {tuple(sorted((index_by_id[atom_id] for atom_id in row['atom_ids']))): row['r0_A'] for row in out['input_reaction_validation']['reference_graph']['IS_reference_bond_lengths_A']}
        _, ids, ads, _ = _signatures(IS)
        if not isinstance(target_pair_atom_ids, (tuple, list)) or len(target_pair_atom_ids) != 2 or any((type(atom_id) is not int or atom_id not in ids for atom_id in target_pair_atom_ids)):
            raise ValueError('INVALID_TARGET_ATOM_IDS')
        index_by_id = {atom_id: index for index, atom_id in enumerate(ids)}
        target = tuple(sorted((index_by_id[atom_id] for atom_id in target_pair_atom_ids)))
        if target not in refs:
            raise ValueError('TARGET_BOND_ABSENT_FROM_SAVED_PARENT')
        parent_edges = set(refs)
        product_edges = parent_edges - {target}
        identity = {i: i for i in ads}
        reference_product_edges = set(_graph(
            FS,
            refs,
            criterion_profile=criterion_profile
        )[0])
        if reference_product_edges != product_edges:
            raise ValueError('SAVED_FS_IS_NOT_EXACT_TARGET_BOND_EDIT')
        _permutations(IS, FS)
        reference_parent = _state(
            IS,
            refs,
            identity,
            criterion_profile=criterion_profile
        )
        reference_products = _state(
            FS,
            refs,
            identity,
            criterion_profile=criterion_profile
        )
        contract = reaction_role_contract(
            reaction_id,
            declared=(input_reaction or {}).get('allowed_parent_roles'),
            equation=(input_reaction or {}).get('requested_formula'),
            requested_direction=requested_direction
        ) if criterion_profile == V3_PROFILE else None
        if contract is not None:
            out['requested_role_contract'] = contract
        out['fragment_formula_multiplicity']['reference'] = {
            'parent': reference_parent['fragment_formula_multiplicity'],
            'products': reference_products['fragment_formula_multiplicity']
        }
        out['reference_subtype'] = _subtype(reference_parent, reference_products)
        maps_minus = _permutations(IS, minus)
        maps_plus = _permutations(IS, plus)
        if len(maps_minus) != len(maps_plus):
            raise ValueError('DOWNHILL_PERMUTATION_CONTEXT_MISMATCH')
        out['graph_evidence'] = {
            'saved_parent_edges_atom_ids': [
                list(sorted((ids[a], ids[b]))) for a,
                b in sorted(parent_edges)
            ],
            'saved_FS_edges_atom_ids': [
                list(sorted((ids[a], ids[b]))) for a,
                b in sorted(reference_product_edges)
            ],
            'target_bond_atom_ids': sorted(target_pair_atom_ids),
            'expected_single_edit': 'delete_target_bond_only',
            'permutation_rule': 'one_element_and_isotope_preserving_map_shared_by_both_downhill_endpoints',
            'reference_bond_and_nonreference_cutoffs': {
                'reference_bond_intact_ratio': RULES['reference_bond_intact_ratio'],
                'nonreference_bond_radius_scale': RULES['nonreference_bond_radius_scale']
            },
            'reference_fragment_roles': {
                'parent': [{
                    'atom_ids': row['mapped_reference_atom_ids'],
                    'role': row['role']
                } for row in reference_parent['fragments']],
                'products': [{
                    'atom_ids': row['mapped_reference_atom_ids'],
                    'role': row['role']
                } for row in reference_products['fragments']]
            }
        }
        candidate_mappings = []
        for mapping in maps_minus:
            state_minus = _state(
                minus,
                refs,
                mapping,
                criterion_profile=criterion_profile
            )
            state_plus = _state(
                plus,
                refs,
                mapping,
                criterion_profile=criterion_profile
            )
            edges_minus = _edge_set(state_minus, ids)
            edges_plus = _edge_set(state_plus, ids)
            for parent_side, product_side in (('minus', 'plus'), ('plus', 'minus')):
                states = {'minus': state_minus, 'plus': state_plus}
                if _edge_set(states[parent_side], ids) != parent_edges or _edge_set(states[product_side], ids) != product_edges:
                    continue
                if states[parent_side]['fragment_formula_multiplicity'] != reference_parent['fragment_formula_multiplicity'] or states[product_side]['fragment_formula_multiplicity'] != reference_products['fragment_formula_multiplicity']:
                    continue
                role_match = _requested_roles_compatible(
                    states[parent_side],
                    states[product_side],
                    contract
                ) if contract is not None else _parent_role_compatible(
                    states[parent_side],
                    reference_parent,
                    criterion_profile
                ) and _role_map(states[product_side]) == _role_map(reference_products)
                score = _mapping_rank(role_match, mapping, ads, parent_side)
                candidate_mappings.append((
                    score,
                    mapping,
                    states,
                    parent_side,
                    product_side
                ))
        if candidate_mappings:
            _, mapping, states, parent_side, product_side = min(
                candidate_mappings,
                key=lambda row: row[0]
            )
            out['orientation'] = {
                'minus': 'parent' if parent_side == 'minus' else 'products',
                'plus': 'parent' if parent_side == 'plus' else 'products'
            }
            out['endpoint_states'] = states
            out['fragment_formula_multiplicity']['observed'] = {
                'minus': states['minus']['fragment_formula_multiplicity'],
                'plus': states['plus']['fragment_formula_multiplicity']
            }
            out['site_channel'] = _site_channel(
                states['minus'],
                states['plus'],
                reference_parent,
                reference_products,
                out['orientation']
            )
            inverse = {reference: actual for actual, reference in mapping.items()}
            out['graph_evidence'].update({
                'consistent_mapping_actual_to_reference_atom_ids': [{
                    'actual_atom_id': ids[i],
                    'reference_atom_id': ids[mapping[i]]
                } for i in ads],
                'actual_bond_deleted_atom_ids': sorted((
                    ids[inverse[target[0]]],
                    ids[inverse[target[1]]]
                )),
                'target_edit_matched_under_one_mapping': True,
                'independent_endpoint_permutations_used': False
            })
            parent_state, product_state = (states[parent_side], states[product_side])
            out['bond_transformation_compatible'] = True
            out['bond_edit_compatible'] = True
            out['observed_subtype'] = _subtype(parent_state, product_state)
            out['channel_subtype'] = out['observed_subtype']
            reference_match = _parent_role_compatible(
                parent_state,
                reference_parent,
                criterion_profile
            ) and _role_map(product_state) == _role_map(reference_products)
            role_match = _requested_roles_compatible(parent_state, product_state, contract) if contract is not None else reference_match
            out['reference_role_compatible'] = reference_match
            out['requested_role_compatible'] = role_match if contract is not None else None
            out['selected_parent_role'] = parent_state['fragments'][0]['role'] if len(parent_state['fragments']) == 1 else None
            out['role_mismatch'] = not role_match
            out['graph_evidence']['reference_fragment_roles_matched'] = reference_match
            out['graph_evidence']['requested_reaction_roles_matched'] = role_match if contract is not None else None
            out['weak_contact_parent'] = criterion_profile in (V2_PROFILE, V3_PROFILE) and any((row['role'] == 'weak_contact' for row in parent_state['fragments']))
            if contract is None and role_match and out['weak_contact_parent']:
                out['observed_subtype'] = out['reference_subtype']
                out['channel_subtype'] = out['reference_subtype']
            ambiguity_states = (parent_state, product_state) if contract is not None else (
                reference_parent,
                reference_products,
                parent_state,
                product_state
            )
            if any((fragment['role'] == 'ambiguous' for state in ambiguity_states for fragment in state['fragments'])):
                out.update(status='unknown', reason='AMBIGUOUS_FRAGMENT_SURFACE_ROLE')
            elif role_match:
                out.update(
                    status='compatible',
                    reason='TARGET_BOND_EDIT_AND_REQUESTED_REACTION_ROLES_MATCH' if contract is not None else 'TARGET_BOND_EDIT_AND_FROZEN_FRAGMENT_ROLES_MATCH'
                )
            else:
                out.update(
                    status='incompatible',
                    channel_subtype='requested_roles_mismatch' if contract is not None else 'role_variant',
                    reason='TARGET_BOND_EDIT_BUT_REQUESTED_ROLES_MISMATCH' if contract is not None else 'TARGET_BOND_EDIT_BUT_FROZEN_FRAGMENT_ROLES_DIFFER'
                )
            return out
        state_minus = _state(minus, refs, identity, criterion_profile=criterion_profile)
        state_plus = _state(plus, refs, identity, criterion_profile=criterion_profile)
        out['endpoint_states'] = {'minus': state_minus, 'plus': state_plus}
        out['fragment_formula_multiplicity']['observed'] = {
            'minus': state_minus['fragment_formula_multiplicity'],
            'plus': state_plus['fragment_formula_multiplicity']
        }
        out['site_channel'] = _site_channel(state_minus, state_plus)
        out['graph_evidence']['target_edit_matched_under_one_mapping'] = False
        out['bond_edit_compatible'] = False
        out['bond_transformation_compatible'] = False
        observed = (_edge_set(state_minus, ids), _edge_set(state_plus, ids))
        if observed[0] == observed[1] == parent_edges:
            roles = [row['role'] for state in (state_minus, state_plus) for row in state['fragments']]
            reason = 'MOLECULAR_ADSORPTION_DESORPTION_NOT_BOND_CLEAVAGE' if 'detached_molecule' in roles and 'surface_contact' in roles else 'SAME_CHEMISTRY_SURFACE_REARRANGEMENT_NOT_TARGET_BOND_EDIT'
            out['channel_subtype'] = 'adsorption_only' if reason.startswith('MOLECULAR_') else 'same_chemistry_rearrangement'
        elif observed[0] == observed[1] == product_edges:
            reason = 'SAME_PRODUCT_CHEMISTRY_REARRANGEMENT_NOT_TARGET_BOND_EDIT'
            out['channel_subtype'] = 'same_product_rearrangement'
        else:
            reason = 'DIFFERENT_BOND_EDIT_OR_NO_CONSISTENT_ATOM_PERMUTATION'
            out['channel_subtype'] = 'different_or_unresolved_bond_edit'
        out.update(status='incompatible', reason=reason)
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
        out.update(status='unknown', reason='CHEMICAL_STATE_UNRESOLVED', error=str(exc))
    return out
