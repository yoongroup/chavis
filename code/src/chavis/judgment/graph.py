from collections import Counter
from itertools import combinations, permutations, product
import math
import numpy as np
from ase.geometry import find_mic
from .policy import NUMERIC_POLICY, RADII, V2_PROFILE, bond_intact, legacy_rules, negative_curvature, surface_role
from .surface_context import selected_surface_element
POLICY_VERSION = 'ni-drm-connection-v1'
RULES = legacy_rules()
PRODUCTS = {
    'D01': ['CH3', 'H'],
    'D02': ['CH2', 'H'],
    'D03': ['CH', 'H'],
    'D04': ['C', 'H'],
    'D05': ['CO', 'O'],
    'D06': ['CO', 'H'],
    'D07': ['H', 'O'],
    'D08': ['HO', 'H'],
    'D09': ['C', 'O'],
    'D10': ['CH', 'O'],
    'D11': ['CHO', 'H'],
    'D12': ['CH2', 'O'],
    'D13': ['H', 'H'],
    'X01': ['CH2O', 'H'],
    'X02': ['CH3', 'O'],
    'N01': ['CHO', 'H'],
    'N02': ['CO', 'H'],
    'N03': ['CH', 'HO'],
    'N04': ['C', 'HO']
}
FORMULAS = {
    'H2': {'H': 2},
    'H2O': {'H': 2, 'O': 1},
    'CO2': {'C': 1, 'O': 2},
    'CH4': {'C': 1, 'H': 4},
    'CH3': {'C': 1, 'H': 3},
    'CH2': {'C': 1, 'H': 2},
    'CH': {'C': 1, 'H': 1},
    'OH': {'O': 1, 'H': 1},
    'CO': {'C': 1, 'O': 1},
    'CHO': {'C': 1, 'H': 1, 'O': 1},
    'CH2O': {'C': 1, 'H': 2, 'O': 1},
    'CH3O': {'C': 1, 'H': 3, 'O': 1},
    'CHOH': {'C': 1, 'H': 2, 'O': 1},
    'COH': {'C': 1, 'H': 1, 'O': 1}
}


def _structure(s):
    if not isinstance(s, dict):
        raise ValueError('MISSING_STRUCTURE')
    p = np.asarray(s['positions'], float)
    cell = np.asarray(s['cell'], float)
    symbols = list(s['symbols'])
    ids = s.get('atom_ids')
    ads = s.get('adsorbate_indices', s.get('meta', {}).get('adsorbate_indices'))
    fixed = s.get('fixed_indices')
    pbc = s.get('pbc')
    if p.shape != (len(symbols), 3) or not np.isfinite(p).all() or cell.shape != (3, 3) or (not np.isfinite(cell).all()) or (not isinstance(ids, list)) or (len(ids) != len(p)) or (len(set(ids)) != len(ids)) or (not isinstance(ads, list)) or (not ads) or (len(set(ads)) != len(ads)) or (not isinstance(fixed, list)) or (not isinstance(pbc, list)):
        raise ValueError('INVALID_STRUCTURE_CONTEXT')
    if any((type(i) is not int or not 0 <= i < len(p) for i in ads + fixed)):
        raise ValueError('INVALID_ATOM_INDICES')
    if any((x not in RADII for x in symbols)):
        raise ValueError('UNSUPPORTED_ELEMENT')
    return (p, cell, symbols, ids, ads, fixed, pbc)


def _distances(s):
    p, cell, _, _, _, _, pbc = _structure(s)
    v = p[:, None, :] - p[None, :, :]
    _, d = find_mic(v.reshape(-1, 3), cell, pbc=pbc)
    return np.asarray(d).reshape(len(p), len(p))


def _formula(symbols):
    count = Counter(symbols)
    return ''.join((e + (str(count[e]) if count[e] > 1 else '') for e in sorted(count)))


def _reference_edges(s, species, *, criterion_profile=V2_PROFILE):
    p, _, symbols, _, ads, _, _ = _structure(s)
    if species not in FORMULAS or Counter((symbols[i] for i in ads)) != Counter(FORMULAS[species]):
        raise ValueError('SPECIES_COMPOSITION_MISMATCH')
    by = {e: [i for i in ads if symbols[i] == e] for e in ('C', 'O', 'H')}
    if species == 'H2':
        edges = [tuple(by['H'])]
    elif species in ('H2O', 'OH'):
        edges = [(by['O'][0], h) for h in by['H']]
    elif species in ('CHOH', 'COH'):
        o = by['O'][0]
        c = by['C'][0]
        distances = _distances(s)
        attached_o = min(by['H'], key=lambda h: distances[o, h])
        if distances[o, attached_o] > NUMERIC_POLICY['reference_input_radius_scale'] * (RADII['O'] + RADII['H']):
            raise ValueError('REFERENCE_OH_BOND_MISSING')
        carbon_h = [h for h in by['H'] if h != attached_o]
        if species == 'CHOH' and len(carbon_h) != 1:
            raise ValueError('REFERENCE_CHOH_MAPPING')
        if species == 'COH' and carbon_h:
            raise ValueError('REFERENCE_COH_MAPPING')
        edges = [(c, o), (o, attached_o)] + [(c, h) for h in carbon_h]
    else:
        c = by['C'][0]
        edges = [(c, i) for i in by['O'] + by['H']]
    d = _distances(s)
    for a, b in edges:
        if not bond_intact(
            d[a, b],
            symbols[a],
            symbols[b],
            nonreference_scale=NUMERIC_POLICY['reference_input_radius_scale'],
            criterion_profile=criterion_profile
        ):
            raise ValueError('REFERENCE_IS_NOT_MOLECULAR_GEOMETRY')
    return {tuple(sorted((a, b))): float(d[a, b]) for a, b in edges}


def _graph(s, refs, *, criterion_profile=V2_PROFILE):
    _, _, symbols, _, ads, _, _ = _structure(s)
    d = _distances(s)
    edges = []
    for a, b in combinations(ads, 2):
        pair = tuple(sorted((a, b)))
        if bond_intact(
            d[a, b],
            symbols[a],
            symbols[b],
            reference_length_A=refs.get(pair),
            criterion_profile=criterion_profile
        ):
            edges.append(pair)
    return (edges, d)


structure_context = _structure
distances = _distances
reference_edges = _reference_edges
molecular_graph = _graph


def validate_molecular_input(structure, species, *, criterion_profile=V2_PROFILE):
    refs = _reference_edges(structure, species, criterion_profile=criterion_profile)
    return {
        'valid': True,
        'species': species,
        'reference_bonds': [
            {'indices': list(pair), 'r0_A': length} for pair,
            length in refs.items()
        ]
    }


def _canonical(s, edges):
    _, _, symbols, _, ads, _, _ = _structure(s)
    groups = [[i for i in ads if symbols[i] == e] for e in sorted(set((symbols[i] for i in ads)))]
    edge_set = set(edges)
    signatures = []
    for perm in product(*(permutations(g) for g in groups)):
        order = sum((list(g) for g in perm), [])
        bits = ''.join((
            '1' if tuple(sorted((a, b))) in edge_set else '0' for a,
            b in combinations(order, 2)
        ))
        signatures.append(''.join((symbols[i] for i in order)) + ':' + bits)
    return min(signatures)


def _endpoint(s, refs, *, criterion_profile=V2_PROFILE):
    p, cell, symbols, ids, ads, fixed, pbc = _structure(s)
    edges, d = _graph(s, refs, criterion_profile=criterion_profile)
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    slab = [i for i in range(len(p)) if i not in ads]
    if len(slab) == 36 and len(fixed) == 18 and all((symbols[i] == 'Ni' for i in slab)):
        metal = 'Ni'
    else:
        metal = selected_surface_element(
            s,
            positions=p,
            cell=cell,
            symbols=symbols,
            adsorbate_indices=ads,
            fixed_indices=fixed,
            pbc=s.get('pbc')
        )
    if normal[2] < 0:
        normal = -normal
    heights = p @ normal
    top = sorted(slab, key=lambda i: heights[i])[-9:]
    plane = float(np.mean(heights[top]))
    adjacency = {i: set() for i in ads}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(ads)
    components = []
    while remaining:
        todo = [min(remaining)]
        members = set(todo)
        while todo:
            for j in sorted(adjacency[todo.pop()] - members):
                members.add(j)
                todo.append(j)
        remaining -= members
        members = sorted(members)
        nearest = [float(min((d[i, j] for j in slab))) for i in members]
        scaled = min((d[i, j] / (RADII[symbols[i]] + RADII[metal]) for i in members for j in slab))
        hmin = float(min((heights[i] - plane for i in members)))
        role = surface_role(
            scaled,
            hmin,
            criterion_profile=criterion_profile,
            intact_parent=set(edges) == set(refs) and len(members) == len(ads)
        )
        component = dict(
            atom_ids=[ids[i] for i in members],
            formula=_formula([symbols[i] for i in members]),
            role=role,
            min_height_A=hmin,
            mean_height_A=float(np.mean(heights[members]) - plane)
        )
        if metal == 'Ni':
            component.update(
                nearest_Ni_A=nearest,
                min_Ni_distance_radius_ratio=float(scaled)
            )
        else:
            component.update(
                surface_element=metal,
                nearest_metal_A=nearest,
                min_metal_distance_radius_ratio=float(scaled)
            )
        components.append(component)
    return dict(
        canonical_graph=_canonical(s, edges),
        edges_atom_ids=[[ids[a], ids[b]] for a, b in edges],
        components=components,
        component_formulas=sorted((c['formula'] for c in components)),
        all_surface_contact=all((c['role'] == 'surface_contact' for c in components))
    )


def _normalize(result):
    r = result.get('report', result)
    art = result.get('artifacts', r.get('artifacts', {}))
    ts = r.get('TS') if isinstance(r.get('TS'), dict) and 'positions' in r['TS'] else art.get('TS')
    dm = r.get('dimer', r.get('ts', r.get('TS', {})))
    fr = r.get('frequencies', {})
    cons = r.get('connections', {})
    if 'results' in cons:
        cons = {x['direction']: {
            **x,
            'structure': art.get('connection_' + x['direction'])
        } for x in cons['results']}
    return (r, ts, dm, fr, cons)


def analyze_candidate(
    result,
    IS,
    FS,
    species,
    reaction_id,
    *,
    requested_direction='dissociation',
    target_pair=None,
    criterion_profile=V2_PROFILE
):
    out = dict(
        policy_version=POLICY_VERSION,
        policy=RULES.copy(),
        reaction_id=reaction_id,
        species=species,
        coverage={'dissociation': False, 'adsorption': False},
        coverage_level='screening_candidate',
        connection_label='unresolved',
        raw_verdict_preserved=True,
        input_molecular_identity_valid=None,
        requested_direction=requested_direction,
        target_pair=target_pair
    )
    try:
        r, ts, dm, fr, cons = _normalize(result)
        modes = fr.get('cm1', fr.get('cm-1'))
        fixed = ts.get('fixed_indices', []) if ts else []
        n = len(ts.get('symbols', [])) if ts else 0
        free = [i for i in range(n) if i not in fixed]
        complete = bool(fr.get('completed', fr.get('complete', False))) and modes is not None and (len(modes) == 3 * len(free)) and (fr.get('indices') == free) and all((math.isfinite(float(x)) for x in modes))
        nimag = sum((float(x) < -NUMERIC_POLICY['imaginary_epsilon_cm1'] for x in modes)) if complete else None
        stationary = dm.get('converged') is True and isinstance(dm.get('fmax'), (int, float)) and (0 <= dm['fmax'] <= NUMERIC_POLICY['ts_fmax_eV_A']) and isinstance(dm.get('curvature'), (int, float)) and negative_curvature(dm['curvature'])
        out['numerical'] = dict(
            stationary=stationary,
            full_free_frequencies=complete,
            imaginary_count=nimag,
            one_imaginary_saddle=bool(stationary and nimag == 1),
            multi_imaginary_candidate=bool(complete and nimag is not None and (nimag > 1))
        )
        out['original_verdict'] = r.get('classification', {}).get('verdict', r.get('validation_status'))
        try:
            refs = _reference_edges(IS, species, criterion_profile=criterion_profile)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            out.update(
                input_molecular_identity_valid=False,
                reason='INVALID_MOLECULAR_INPUT',
                error=str(exc)
            )
            return out
        out['input_molecular_identity_valid'] = True
        if not all((isinstance(cons.get(k), dict) and cons[k].get('structure') for k in ('minus', 'plus'))):
            out['reason'] = 'DOWNHILL_NOT_AVAILABLE'
            return out
        for k in ('minus', 'plus'):
            e = cons[k]
            if e.get('converged') is not True or not isinstance(e.get('fmax'), (int, float)) or (not 0 <= e['fmax'] <= NUMERIC_POLICY['endpoint_fmax_eV_A']):
                out['reason'] = 'DOWNHILL_NOT_CONVERGED'
                return out
        context = _structure(IS)
        for s in [FS, ts, cons['minus']['structure'], cons['plus']['structure']]:
            c = _structure(s)
            if c[2:6] != context[2:6] or c[6] != context[6] or (not np.allclose(
                c[1],
                context[1],
                atol=NUMERIC_POLICY['cell_context_atol_A'],
                rtol=0
            )) or (not np.allclose(
                c[0][context[5]],
                context[0][context[5]],
                atol=NUMERIC_POLICY['fixed_position_context_atol_A'],
                rtol=0
            )):
                raise ValueError('ENDPOINT_CONTEXT_MISMATCH')
        ref_graph = _canonical(IS, list(refs))
        ends = {k: _endpoint(
            cons[k]['structure'],
            refs,
            criterion_profile=criterion_profile
        ) for k in ('minus', 'plus')}
        _, _, _, ids, _, _, _ = _structure(IS)
        if not isinstance(target_pair, (list, tuple)) or len(target_pair) != 2:
            raise ValueError('TARGET_PAIR_MISSING')
        id_to_index = {atom_id: i for i, atom_id in enumerate(ids)}
        try:
            target_edge = tuple(sorted((
                id_to_index[target_pair[0]],
                id_to_index[target_pair[1]]
            )))
        except KeyError as exc:
            raise ValueError('TARGET_ATOM_ID_MISSING') from exc
        if target_edge not in refs:
            raise ValueError('TARGET_EDGE_NOT_IN_REFERENCE_PARENT')
        expected_product_graph = _canonical(
            IS,
            [edge for edge in refs if edge != target_edge]
        )
        out['expected_product_canonical_graph'] = expected_product_graph
        out['target_edge_atom_ids'] = list(target_pair)
        for e in ends.values():
            e['intact_parent'] = e['canonical_graph'] == ref_graph
        out['endpoints'] = ends
        out['downhill_both_converged'] = True
        a, b = (ends['minus'], ends['plus'])
        target = sorted(PRODUCTS.get(reaction_id, []))

        def target_product_graph(e):
            return bool(target) and e['component_formulas'] == target and (e['canonical_graph'] == expected_product_graph) and (not e['intact_parent'])

        def target_products(e):
            return target_product_graph(e) and e['all_surface_contact']
        graph_transition = a['intact_parent'] and target_product_graph(b) or (b['intact_parent'] and target_product_graph(a))
        out['target_fragment_graph_transition'] = bool(graph_transition)
        out['target_graph_single_imaginary_connection'] = bool(graph_transition and stationary and (nimag == 1))

        def parent_attached(e):
            return e['intact_parent'] and len(e['components']) == 1 and (e['components'][0]['role'] in ('surface_contact', 'weak_contact'))
        diss = parent_attached(a) and target_products(b) or (parent_attached(b) and target_products(a))
        ads = a['intact_parent'] and b['intact_parent'] and (len(a['components']) == len(b['components']) == 1) and ({
            a['components'][0]['role'],
            b['components'][0]['role']
        } == {'surface_contact', 'detached_molecule'})
        direct = a['intact_parent'] and len(a['components']) == 1 and (a['components'][0]['role'] == 'detached_molecule') and target_products(b) or (b['intact_parent'] and len(b['components']) == 1 and (b['components'][0]['role'] == 'detached_molecule') and target_products(a))
        same_graph = a['canonical_graph'] == b['canonical_graph']
        label = 'dissociation_recombination' if diss else 'molecular_adsorption_desorption' if ads else 'direct_dissociative_adsorption' if direct else 'target_bond_change_surface_role_unresolved' if graph_transition else 'surface_rearrangement_or_same_basin' if same_graph and a['all_surface_contact'] and b['all_surface_contact'] else 'other_connection'
        out['connection_label'] = label
        out['coverage'] = {
            'dissociation': bool(diss and stationary and (nimag == 1)),
            'adsorption': bool(ads and stationary and (nimag == 1))
        }
        out['target_requested_connection'] = bool(diss and stationary and (nimag == 1))
        out['search_direction'] = 'computational_cleavage' if requested_direction == 'association' else 'requested_forward'
        out['forward_barrier_reference'] = 'computed_FS_energy' if requested_direction == 'association' else 'computed_IS_energy'
        out['reason'] = 'CLASSIFIED_GEOMETRIC_SCREENING'
        out['limitations'] = [
            'Surface contact/detachment is geometric screening, not gas-asymptote proof.',
            'Canonical chemical graph permits equal-element swaps; spatial symmetry remains separate.',
            'Same graph is not proof of same adsorption basin or same TS.'
        ]
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        out['reason'] = 'ANALYSIS_UNRESOLVED'
        out['error'] = str(exc)
    return out


def strict_is_fs_confirmation(
    result,
    IS,
    FS,
    symmetry,
    *,
    species,
    match_rms_A,
    criterion_profile=V2_PROFILE
):
    out = dict(
        status='unconfirmed',
        criterion='one_imaginary_and_both_downhill_match_saved_IS_FS',
        match_rms_A=float(match_rms_A),
        pairing=None,
        comparison_method=None
    )
    try:
        r, ts, dm, fr, cons = _normalize(result)
        if not isinstance(ts, dict):
            out['reason'] = 'TS_MISSING'
            return out
        if dm.get('converged') is not True or not isinstance(dm.get('fmax'), (int, float)) or (not math.isfinite(dm['fmax'])) or (not 0 <= dm['fmax'] <= NUMERIC_POLICY['ts_fmax_eV_A']) or (not isinstance(dm.get('curvature'), (int, float))) or (not math.isfinite(dm['curvature'])) or (not negative_curvature(dm['curvature'])):
            out['reason'] = 'TS_NONCONVERGED'
            return out
        modes = fr.get('cm1', fr.get('cm-1'))
        fixed = ts.get('fixed_indices', [])
        free = [i for i in range(len(ts['symbols'])) if i not in fixed]
        if not fr.get('completed', fr.get('complete', False)) or modes is None or len(modes) != 3 * len(free) or (fr.get('indices') != free) or any((not math.isfinite(float(x)) for x in modes)):
            out['reason'] = 'FREQUENCY_MISSING_OR_INCOMPLETE'
            return out
        nimag = sum((float(x) < -NUMERIC_POLICY['imaginary_epsilon_cm1'] for x in modes))
        out['imaginary_count'] = nimag
        if nimag != 1:
            out['reason'] = 'MULTIPLE_IMAGINARY' if nimag > 1 else 'ZERO_IMAGINARY'
            return out
        for direction in ('minus', 'plus'):
            row = cons.get(direction)
            if not isinstance(row, dict) or not row.get('structure'):
                out['reason'] = 'DOWNHILL_MISSING'
                return out
            if row.get('converged') is not True or not isinstance(row.get('fmax'), (int, float)) or (not math.isfinite(row['fmax'])) or (not 0 <= row['fmax'] <= NUMERIC_POLICY['endpoint_fmax_eV_A']):
                out['reason'] = 'DOWNHILL_NONCONVERGED'
                return out
        context = _structure(IS)
        for structure in (
            FS,
            ts,
            cons['minus']['structure'],
            cons['plus']['structure']
        ):
            current = _structure(structure)
            if current[2:6] != context[2:6] or current[6] != context[6] or (not np.allclose(
                current[1],
                context[1],
                atol=NUMERIC_POLICY['cell_context_atol_A'],
                rtol=0
            )) or (not np.allclose(
                current[0][context[5]],
                context[0][context[5]],
                atol=NUMERIC_POLICY['fixed_position_context_atol_A'],
                rtol=0
            )):
                out['reason'] = 'ENDPOINT_CONTEXT_MISMATCH'
                return out
        refs = _reference_edges(IS, species, criterion_profile=criterion_profile)
        ref_edges = {
            'IS': set(_graph(IS, refs, criterion_profile=criterion_profile)[0]),
            'FS': set(_graph(FS, refs, criterion_profile=criterion_profile)[0])
        }
        arrival_edges = {d: set(_graph(
            cons[d]['structure'],
            refs,
            criterion_profile=criterion_profile
        )[0]) for d in ('minus', 'plus')}
        raw_rows = {x.get('direction'): x for x in r.get('connections', {}).get('results', []) if isinstance(x, dict)}
        for minus_end, plus_end in (('IS', 'FS'), ('FS', 'IS')):
            if arrival_edges['minus'] != ref_edges[minus_end] or arrival_edges['plus'] != ref_edges[plus_end]:
                continue

            def raw_match(direction, end):
                m = raw_rows.get(direction, {}).get('matches', {}).get(end, {})
                return m.get('edges_match') is True and isinstance(m.get('ads_rms_A'), (int, float)) and (m['ads_rms_A'] <= match_rms_A) and isinstance(m.get('free_rms_A'), (int, float)) and (m['free_rms_A'] <= match_rms_A)
            if raw_match('minus', minus_end) and raw_match('plus', plus_end):
                out.update(
                    status='confirmed',
                    reason='EXACT_RAW_IS_FS_MATCH',
                    pairing={'minus': minus_end, 'plus': plus_end},
                    comparison_method='raw_ID_preserving_MIC_RMS_and_exact_edges'
                )
                return out
            if symmetry.get('status') == 'complete':
                for mode in ('rotation', 'mirror'):
                    comparisons = symmetry.get(mode, {}).get('comparisons', {})
                    minus = comparisons.get(minus_end + '|connection_minus', {})
                    plus = comparisons.get(plus_end + '|connection_plus', {})
                    if isinstance(minus.get('score_A'), (int, float)) and minus['score_A'] <= match_rms_A and isinstance(plus.get('score_A'), (int, float)) and (plus['score_A'] <= match_rms_A):
                        out.update(
                            status='confirmed',
                            reason='SAVED_SYMMETRY_IS_FS_MATCH',
                            pairing={'minus': minus_end, 'plus': plus_end},
                            comparison_method=mode,
                            comparison_scores_A={
                                'minus': minus['score_A'],
                                'plus': plus['score_A']
                            },
                            symmetry_scope=symmetry.get('scope')
                        )
                        return out
        out['reason'] = 'IS_FS_MATCH_UNCONFIRMED'
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        out['reason'] = 'STRICT_EVALUATION_UNRESOLVED'
        out['error'] = str(exc)
    return out
