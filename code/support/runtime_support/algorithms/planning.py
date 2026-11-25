from __future__ import annotations
import copy
from collections import Counter
from geometry import validate
from workbench_tools import components, component
from .catalog import validate_reaction
from .identity import adsorbates, graph, evidence_map, graph_maps, map_slab, MappingLimit


def fragment_compositions(s, g):
    return sorted((tuple(sorted(Counter((s['symbols'][i] for i in ids)).items())) for ids in components(g)))


def select_parent(reactant, product, manual_side=None, params=None):
    a, b = (validate(reactant), validate(product))
    if Counter(a['symbols']) != Counter(b['symbols']):
        raise ValueError('Reactant/product element counts differ. Identical atom sets are required.')
    p = params or {}
    aa, ab = (adsorbates(a), adsorbates(b))
    ga = graph(a, aa, p.get('reactant_params', p))
    gb = graph(b, ab, p.get('product_params', p))
    counts = {'reactant': len(components(ga)), 'product': len(components(gb))}
    if manual_side not in (None, '', 'reactant', 'product'):
        raise ValueError('Parent must be reactant or product.')
    if counts['reactant'] == counts['product']:
        if not manual_side:
            return {
                'status': 'manual_parent_required',
                'counts': counts,
                'note': 'Fragment counts are equal. Select the parent explicitly; bonding is not inferred automatically.'
            }
        side, reason = (manual_side, 'explicit selection (equal fragment counts)')
    else:
        side = min(counts, key=counts.get)
        if manual_side and manual_side != side:
            raise ValueError('The endpoint with more fragments cannot be the parent.')
        reason = 'endpoint with fewer fragments'
    parent, other = (a, b) if side == 'reactant' else (b, a)
    return {
        'status': 'ready',
        'parent_side': side,
        'parent': parent,
        'opposite': other,
        'counts': counts,
        'selection_reason': reason,
        'manual_reviewed': False
    }


def difference(parent_graph, opposite_graph, mapping):
    if set(mapping) != set(parent_graph) or set(mapping.values()) != set(opposite_graph):
        raise ValueError('Adsorbate atom correspondence is not a complete one-to-one mapping.')
    inverse = {j: i for i, j in mapping.items()}
    a = {tuple(sorted((i, j))) for i in parent_graph for j in parent_graph[i]}
    b = {tuple(sorted((inverse[i], inverse[j]))) for i in opposite_graph for j in opposite_graph[i]}
    return {
        'break': [list(x) for x in sorted(a - b)],
        'form': [list(x) for x in sorted(b - a)]
    }


def pair_edits(parent, opposite, placement, params=None):
    p = params or {}
    aa, ab = (adsorbates(parent), adsorbates(opposite))
    ga = graph(parent, aa, p)
    gb = graph(
        opposite,
        ab,
        {**p, 'bond_overrides': p.get('opposite_bond_overrides', {})}
    )
    known = evidence_map(parent, opposite, p.get('mapping_confirmed') is True)
    if known:
        fullmap, source = known
        amap = {i: fullmap[i] for i in aa}
        edits = difference(ga, gb, amap)
        choices = [{'edits': edits, 'mapping': fullmap, 'source': source}]
    else:
        slab_map = map_slab(parent, opposite, aa, ab)
        choices = []
        edges = [(i, j) for i in ga for j in sorted(ga[i]) if i < j]
        candidates = [None] if placement == 'normal_lift' else edges
        for edge in candidates:
            cut = {i: set(js) for i, js in ga.items()}
            if edge:
                i, j = edge
                cut[i].remove(j)
                cut[j].remove(i)
                if j in component(cut, i):
                    continue
            for m in graph_maps(parent, cut, opposite, gb, p.get('max_mappings', 256)):
                choices.append({
                    'edits': difference(ga, gb, m),
                    'mapping': {**slab_map, **m},
                    'source': 'graph-inferred-unreviewed'
                })
                if len(choices) > int(p.get('max_mappings', 256)):
                    raise MappingLimit('R/P correspondence limit exceeded. Stopping without truncation; specify the reaction bond explicitly.')
    valid = []
    for choice in choices:
        e = choice['edits']
        if placement == 'normal_lift':
            okay = not e['break'] and (not e['form'])
        else:
            okay = len(e['break']) == 1 and (not e['form'])
        if okay:
            valid.append(choice)
    if not valid:
        raise ValueError('Transformation is neither single-bond cleavage nor intact detachment. Transfer and rearrangement are unsupported.')
    return valid


def directed_edits(generation_edits, parent_side):
    if parent_side not in ('reactant', 'product'):
        raise ValueError('Specify the parent side of the reaction.')
    e = copy.deepcopy(generation_edits)
    return e if parent_side == 'reactant' else {
        'break': e.get('form', []),
        'form': e.get('break', [])
    }


def catalog_choices(parent, reaction, params=None):
    p = params or {}
    r = validate_reaction(reaction)
    ads = adsorbates(parent, p)
    g = graph(parent, ads, p)
    observed = fragment_compositions(parent, g)
    if r['category'] == 'ADSORPTION':
        if observed != sorted(r['products']):
            raise ValueError('Selected parent adsorbate differs from the reaction molecule. Check adsorbate atom selection.')
        if p.get('parent_side') != 'product' or p.get('bound_parent_confirmed') is not True:
            raise ValueError('Adsorption has equal fragment counts. Select and confirm the adsorbed product parent.')
        lead = ads[0]
        return [{
            'placement': 'normal_lift',
            'parent_side': 'product',
            'anchor': lead,
            'moving': [lead] + sorted(set(ads) - {lead}),
            'generation_edits': {'break': [], 'form': []},
            'adsorbate': ads,
            'selection_reason': 'adsorbed product explicitly confirmed as parent',
            'atom_mapping_source': 'same-parent',
            'mapping_alternative': 1
        }]
    side = 'reactant' if len(r['reactants']) < len(r['products']) else 'product'
    expected = r['reactants'] if side == 'reactant' else r['products']
    target = r['products'] if side == 'reactant' else r['reactants']
    if observed != sorted(expected):
        raise ValueError('Parent composition or components differ from the equation. Check adsorbate selection excluding the slab and its connectivity.')
    choices = []
    selected_cuts = p.get('generation_edits', {}).get('break', [])
    for i in ads:
        for j in sorted(g[i]):
            if j <= i or (selected_cuts and {i, j} not in [set(x) for x in selected_cuts]):
                continue
            cut = {k: set(v) for k, v in g.items()}
            cut[i].remove(j)
            cut[j].remove(i)
            if j in component(cut, i) or fragment_compositions(parent, cut) != sorted(target):
                continue
            ai, bj = (component(cut, i), component(cut, j))
            directions = [(i, j), (j, i)] if len(ai) == len(bj) or p.get('move_both_sides') else [(i, j) if len(bj) < len(ai) else (j, i)]
            for anchor, lead in directions:
                choices.append({
                    'placement': 'surface_sites',
                    'parent_side': side,
                    'anchor': anchor,
                    'moving': [lead] + sorted(component(cut, lead) - {lead}),
                    'generation_edits': {'break': [[i, j]], 'form': []},
                    'adsorbate': ads,
                    'selection_reason': 'reaction equation and parent fragment composition verified',
                    'atom_mapping_source': 'same-parent',
                    'mapping_alternative': len(choices) + 1
                })
    if not choices:
        raise ValueError('No bond splits into the two fragments required by this equation.')
    return choices
