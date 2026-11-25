from __future__ import annotations
import copy
import hashlib
import json
import math
from geometry import validate, parse_indices
from surface_engine import generate_surface
from .identity import stamp_import, inherit_identity, adsorbates, graph, recipe_maps
from .planning import catalog_choices, select_parent, pair_edits, directed_edits
from workbench_tools import component


def signature(parent):
    s = validate(parent)
    fields = ('symbols', 'positions', 'atom_ids', 'cell', 'pbc', 'fixed_indices')
    return hashlib.sha256(json.dumps(
        {k: s.get(k) for k in fields},
        sort_keys=True,
        allow_nan=False
    ).encode()).hexdigest()


def rotation_values(params):
    value = params.get('rotations_deg')
    if value is None or value == '':
        value = [params.get('rotation', 0)]
    elif isinstance(value, str):
        value = [x.strip() for x in value.split(',') if x.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError('Rotation angles must be a numeric list.')
    angles = [float(x) for x in value]
    if not 1 <= len(angles) <= 36 or any((not math.isfinite(x) for x in angles)):
        raise ValueError('Supply 1~36 finite rotation angles.')
    return angles


def manual_choice(parent, params):
    p = copy.deepcopy(params)
    mode = p.get('mode', 'cleavage')
    if mode in ('association', 'transfer', 'rearrangement'):
        raise ValueError('Generation separates a bonded parent. Select the bonded endpoint as parent. Automated transfer and rearrangement are excluded; manual editing remains available.')
    if mode not in ('cleavage', 'desorption', 'adsorption'):
        raise ValueError('Unsupported separation operation.')
    side = p.get('parent_side', 'reactant')
    if side not in ('reactant', 'product'):
        raise ValueError('Specify whether the parent is the reactant or product.')
    if mode == 'adsorption':
        if side != 'product' or p.get('bound_parent_confirmed') is not True:
            raise ValueError('Adsorption generation detaches the molecule from an explicitly confirmed adsorbed product parent.')
        mode = 'desorption'
    p['mode'] = mode
    if 'generation_edits' in p:
        p['reaction_edits'] = copy.deepcopy(p['generation_edits'])
    if mode == 'desorption' and any((p.get('reaction_edits', {}).get(k) for k in ('break', 'form'))):
        raise ValueError('Intact molecule detachment cannot include internal bond edits. Reset the edits before generating.')
    return (p, side, p.get('atom_mapping_source', 'same-parent'))


def prepare_pair(reactant, product, params=None, manual_side=None):
    p = copy.deepcopy(params or {})
    selection = select_parent(reactant, product, manual_side, p)
    if selection['status'] != 'ready':
        return selection
    parent, opposite, side = (
        selection['parent'],
        selection['opposite'],
        selection['parent_side']
    )
    placement = p.get('placement', 'surface_sites')
    if p.get('reaction', {}).get('category') == 'ADSORPTION':
        placement = 'normal_lift'
    choices = []
    for record in pair_edits(parent, opposite, placement, p):
        edits = record['edits']
        ads = adsorbates(parent)
        g = graph(parent, ads, p)
        if placement == 'normal_lift':
            if p.get('bound_parent_confirmed') is not True:
                raise ValueError('Confirm that the intact parent molecule is adsorbed.')
            if set(component(g, ads[0])) != set(ads):
                raise ValueError('Specify one intact molecule to detach.')
            pairs = [(ads[0], ads[0], list(ads))]
        else:
            i, j = edits['break'][0]
            cut = {k: set(v) for k, v in g.items()}
            cut[i].remove(j)
            cut[j].remove(i)
            a, b = (component(cut, i), component(cut, j))
            if j in a:
                raise ValueError('Cannot represent this transformation as single-bond cleavage.')
            dirs = [(i, j), (j, i)] if len(a) == len(b) or p.get('move_both_sides') else [(i, j)] if len(b) < len(a) else [(j, i)]
            pairs = [(x, y, [y] + sorted(component(cut, y) - {y})) for x, y in dirs]
        for anchor, lead, moving in pairs:
            choices.append({
                'anchor': anchor,
                'moving': moving,
                'adsorbate': ads,
                'mode': 'desorption' if placement == 'normal_lift' else 'cleavage',
                'generation_edits': edits,
                'parent_side': side,
                'atom_mapping_source': record['source'],
                'endpoint_atom_map': record['mapping']
            })
    if p.get('reaction'):
        catalog_choices(parent, p['reaction'], {**p, 'parent_side': side})
    return {
        **selection,
        'choices': choices,
        'plan_signature': signature(parent),
        'plan_graph_params': {k: copy.deepcopy(p.get(k)) for k in (
            'bond_scale',
            'graph_backend',
            'bond_overrides'
        )},
        'manual_reviewed': False
    }


def annotate(candidate, parent, p, side, name, source, alternative=1):
    c = copy.deepcopy(candidate)
    c['input_parent'] = inherit_identity(parent, c.get('input_parent', parent))
    c['structure'] = inherit_identity(parent, c['structure'])
    detail = c.setdefault('detail', {})
    edits = copy.deepcopy(detail.get('reaction_edits') or p.get('generation_edits') or p.get('reaction_edits') or {'break': [], 'form': []})
    c.update(
        parent_side=side,
        generation_edits=edits,
        reaction_edits=directed_edits(edits, side),
        path_direction='parent_to_opposite',
        atom_mapping_source=source,
        mapping_reviewed=False
    )
    detail.update(generation_edits=copy.deepcopy(edits), parent_side=side)
    detail['reaction_edits'] = copy.deepcopy(edits)
    moving = detail.get(
        'moving',
        parse_indices(p.get('moving', []), len(parent['symbols']))
    )
    words = ', '.join((
        f"{parent['symbols'][i]}({i})–{parent['symbols'][j]}({j})" for i,
        j in edits.get('break', [])
    ))
    site = detail.get('site', '')
    label = {
        'top': 'top',
        'bridge': 'bridge',
        'fcc': 'fcc hollow',
        'hcp': 'hcp hollow',
        'hollow_unknown': 'hollow (unclassified)',
        'normal': 'outward surface normal'
    }.get(site, site or 'position unavailable')
    trace = {
        'parent_name': name,
        'source_rotation_deg': parent.get('meta', {}).get('azimuth_deg'),
        'applied_rotation_deg': p.get('rotation', 0),
        'reaction': p.get('reaction', {}).get('equation', 'explicit selection'),
        'operation_text': words + ' bond cleavage' if words else 'intact molecule displacement away from surface',
        'site_text': label,
        'moving_atoms': [f"{parent['symbols'][i]}({i})" for i in moving],
        'radius_A': detail.get('distance_A'),
        'height_A': detail.get('height_A'),
        'lift_A': detail.get('displacement_A'),
        'cell_shift': detail.get('cell_shift', [0, 0]),
        'target_position_A': c['structure']['positions'][moving[0]][:] if moving else None,
        'mapping_text': {
            'same-parent': 'preserve parent atom indices and order',
            'manual-confirmation': 'use confirmed atom correspondence between parents',
            'shared-lineage': 'use common-lineage atom correspondence',
            'shared-namespace': 'use common-index atom correspondence'
        }.get(source, 'connectivity-inferred atom correspondence requires review'),
        'mapping_alternative': alternative,
        'manual_reviewed': False
    }
    c['provenance'] = trace
    c['structure'].setdefault('meta', {})['generation_provenance'] = copy.deepcopy(trace)
    c['sourceParams'] = copy.deepcopy(p)
    return c


def generate(parent, params):
    p = copy.deepcopy(params)
    s = stamp_import(parent, p.get('parent_name', ''))
    s['adsorbate_indices'] = adsorbates(s, p)
    s.setdefault('meta', {})['adsorbate_indices'] = s['adsorbate_indices'][:]
    kind = p.get('algorithm_mode', 'manual')
    if kind == 'catalog':
        if not p.get('reaction'):
            raise ValueError('Select the reaction for automated search. Manual generation does not require a reaction database.')
        automatic = copy.deepcopy(p)
        automatic.pop('generation_edits', None)
        choices = catalog_choices(s, p['reaction'], automatic)
        choices = [{
            **x,
            'mode': 'desorption' if x['placement'] == 'normal_lift' else 'cleavage',
            'fragment_policy': 'auto'
        } for x in choices]
    elif kind == 'pair':
        plan = p.get('pair_plan') or {}
        current_graph = {k: copy.deepcopy(p.get(k)) for k in (
            'bond_scale',
            'graph_backend',
            'bond_overrides'
        )}
        if plan.get('plan_signature') != signature(s) or plan.get('plan_graph_params') != current_graph:
            raise ValueError('Parent or connectivity settings changed after R/P analysis. Repeat the analysis.')
        choices = plan.get('choices') or []
    elif kind == 'manual':
        choices = [{}]
    else:
        raise ValueError('Unknown generation method.')
    angles = rotation_values(p)
    if len(choices) * len(angles) > 1000:
        raise ValueError('Bond/angle combinations exceed 1000. Stopping without truncation.')
    out = []
    limit = int(p.get('max_candidates', 5000))
    if not 1 <= limit <= 10000:
        raise ValueError('Candidate limit must be 1~10000.')
    for index, choice in enumerate(choices, 1):
        for angle in angles:
            q = {**p, **choice, 'rotation': angle}
            q, side, source = manual_choice(s, q)
            values = generate_surface(s, q)
            if len(out) + len(values) > limit:
                raise ValueError(f'Candidate {limit} limit exceeded. Partial results are not returned as complete.')
            out.extend((annotate(
                c,
                s,
                q,
                side,
                p.get('parent_name') or s.get('meta', {}).get('source_name', 'explicit parent'),
                source,
                index
            ) for c in values))
    if not out:
        raise ValueError('No candidates to generate.')
    return out


def map_recipe(source, target, params):
    p = copy.deepcopy(params)
    p.pop('mapping_alternatives', None)
    a, b = (validate(source), validate(target))
    if p.get('algorithm_mode') == 'catalog':
        p['adsorbate'] = adsorbates(b)
        p['parent_name'] = b.get('meta', {}).get('source_name', 'target parent')
        return {
            'params': p,
            'alternatives': [],
            'note': 'Search reaction bonds independently for each parent.'
        }
    if p.get('algorithm_mode') == 'pair':
        raise ValueError('The R/P plan applies only to its analyzed parent. Use reaction database search for multiple parents.')
    if a == b:
        a = stamp_import(a)
        a['adsorbate_indices'] = adsorbates(a, p)
        b = copy.deepcopy(a)
    choices = recipe_maps(a, b, p)
    for record in choices:
        record['params']['parent_name'] = b.get('meta', {}).get('source_name', 'target parent')
    q = copy.deepcopy(choices[0]['params'])
    if len(choices) > 1:
        q['mapping_alternatives'] = [copy.deepcopy(x['params']) for x in choices]
        q['mapping_target_signature'] = signature(b)
    return {
        'params': q,
        'alternatives': choices,
        'note': f'Possible atom correspondences: {len(choices)}. Generate every correspondence and check duplicates separately.'
    }
