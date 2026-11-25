from __future__ import annotations
import copy
import uuid
from collections import Counter
from geometry import validate, parse_indices, norm
from workbench_tools import connectivity, displacement


class MappingLimit(ValueError):
    pass


def stamp_import(structure, source_name=''):
    s = validate(structure)
    meta = s.setdefault('meta', {})
    identity = copy.deepcopy(meta.get('identity') or {})
    identity.setdefault('lineage_id', str(uuid.uuid4()))
    identity.setdefault('id_namespace', None)
    identity.setdefault('origin', meta.get('atom_id_source', 'row-order-unverified'))
    identity.setdefault('manual_reviewed', False)
    identity.setdefault(
        'note',
        'Atom correspondence between independent files is unreviewed. File-local indices do not establish global identity.'
    )
    meta['identity'] = identity
    meta.setdefault('source_name', source_name)
    return s


def inherit_identity(parent, child):
    a, b = (validate(parent), validate(child))
    if a['symbols'] != b['symbols'] or a['atom_ids'] != b['atom_ids']:
        raise ValueError('Atom order or indices changed in the parent-derived structure.')
    b.setdefault('meta', {})['identity'] = copy.deepcopy(a.get('meta', {}).get('identity', {}))
    b['meta']['identity']['derivation'] = 'same-parent-coordinate-only'
    b['meta']['identity'].setdefault('manual_reviewed', False)
    return b


def adsorbates(s, params=None):
    p = params or {}
    value = p.get(
        'adsorbate',
        s.get('adsorbate_indices', s.get('meta', {}).get('adsorbate_indices'))
    )
    if value is None:
        raise ValueError('Specify adsorbate atom indices first. Remaining atoms are treated as slab atoms.')
    ids = parse_indices(value, len(s['symbols']))
    if not ids:
        raise ValueError('No adsorbate atoms selected.')
    return ids


def graph(s, ids, params=None):
    p = params or {}
    return connectivity(
        s,
        ids,
        float(p.get('bond_scale', 1.2)),
        p.get('graph_backend', 'auto'),
        p.get('bond_overrides')
    )[0]


def evidence_map(source, target, explicitly_confirmed=False):
    a, b = (validate(source), validate(target))
    ia = a.get('meta', {}).get('identity', {})
    ib = b.get('meta', {}).get('identity', {})
    same_lineage = bool(ia.get('lineage_id')) and ia.get('lineage_id') == ib.get('lineage_id')
    same_namespace = bool(ia.get('id_namespace')) and ia.get('id_namespace') == ib.get('id_namespace')
    if not (same_lineage or same_namespace or explicitly_confirmed):
        return None
    lookup = {aid: j for j, aid in enumerate(b['atom_ids'])}
    if len(a['symbols']) != len(b['symbols']) or set(a['atom_ids']) != set(b['atom_ids']):
        raise ValueError('Common-lineage structures have different atom index sets. Review lineage records.')
    mapping = {i: lookup[aid] for i, aid in enumerate(a['atom_ids'])}
    if any((a['symbols'][i] != b['symbols'][j] for i, j in mapping.items())):
        raise ValueError('Recorded corresponding atoms have different elements.')
    return (
        mapping,
        'manual-confirmation' if explicitly_confirmed else 'shared-lineage' if same_lineage else 'shared-namespace'
    )


def graph_maps(a, ga, b, gb, limit=256):
    try:
        import networkx as nx
    except ImportError as exc:
        raise ValueError('Atom correspondence generation requires networkx. Install requirements-algorithms.txt.') from exc
    fixed_a, fixed_b = (
        set(a.get('fixed_indices', [])),
        set(b.get('fixed_indices', []))
    )

    def build(s, g, fixed):
        result = nx.Graph()
        moments = s.get('initial_magmoms')
        for i in g:
            if moments is not None and isinstance(moments[i], (list, tuple)):
                raise ValueError('Atom correspondence with noncollinear magnetic moments is unsupported.')
            label = (
                s['symbols'][i],
                i in fixed,
                round(float(moments[i]), 6) if moments is not None else None
            )
            result.add_node(i, label=label)
        result.add_edges_from(((i, j) for i in g for j in g[i]))
        return result
    aa, bb = (build(a, ga, fixed_a), build(b, gb, fixed_b))
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        aa,
        bb,
        node_match=lambda x,
        y: x['label'] == y['label']
    )
    out = []
    for mapping in matcher.isomorphisms_iter():
        if len(out) >= int(limit):
            raise MappingLimit(f'Atom correspondence candidates exceed {limit}. No automatic truncation was performed. Specify the reaction center explicitly.')
        out.append({int(i): int(j) for i, j in mapping.items()})
    return out


def map_slab(a, b, ads_a, ads_b, tolerance=0.15):
    if a['pbc'] != b['pbc'] or any((
        abs(x - y) > 1e-06 for ra,
        rb in zip(a['cell'], b['cell']) for x,
        y in zip(ra, rb)
    )):
        raise ValueError('Parent cell/PBC differ. Align their surface coordinate systems first.')
    sa = [i for i in range(len(a['symbols'])) if i not in set(ads_a)]
    sb = [i for i in range(len(b['symbols'])) if i not in set(ads_b)]
    if len(sa) != len(sb):
        raise ValueError('Parent slab atom counts differ.')
    fixed_a, fixed_b = (
        set(a.get('fixed_indices', [])),
        set(b.get('fixed_indices', []))
    )
    mapping = {}
    for i in sa:
        hits = [j for j in sb if a['symbols'][i] == b['symbols'][j] and (i in fixed_a) == (j in fixed_b) and (norm(displacement(a['positions'][i], b['positions'][j], a)) <= tolerance)]
        if len(hits) != 1:
            raise ValueError('Slab atom correspondence is ambiguous. Common lineage or explicit correspondence is required.')
        mapping[i] = hits[0]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Slab atom correspondence is not one-to-one.')
    return mapping


def recipe_maps(source, target, params):
    a, b = (validate(source), validate(target))
    p = copy.deepcopy(params)
    aa = adsorbates(a, p)
    ab = adsorbates(
        b,
        {'adsorbate': p['target_adsorbate']} if 'target_adsorbate' in p else None
    )
    ga = graph(a, aa, p)
    gb = graph(b, ab, {**p, 'bond_overrides': p.get('target_bond_overrides', {})})
    evidence = evidence_map(a, b, p.get('mapping_confirmed') is True)
    if evidence:
        mapping, source_label = evidence
        if {mapping[i] for i in aa} != set(ab) or any(({mapping[j] for j in ga[i]} != gb[mapping[i]] for i in aa)):
            raise ValueError('Existing atom correspondence conflicts with connectivity and will not be overwritten.')
        mappings = [mapping]
    else:
        if Counter(a['symbols']) != Counter(b['symbols']):
            raise ValueError('Parent compositions differ. Select separate reactions for different species.')
        slab_map = map_slab(a, b, aa, ab)
        mappings = [{**slab_map, **m} for m in graph_maps(
            a,
            ga,
            b,
            gb,
            p.get('max_mappings', 256)
        )]
        source_label = 'graph-inferred-unreviewed'
    if not mappings:
        raise ValueError('No connectivity-preserving atom correspondence found.')
    out = []
    for index, mapping in enumerate(mappings, 1):
        q = copy.deepcopy(p)
        for key in ('adsorbate', 'moving', 'surface_indices'):
            value = aa if key == 'adsorbate' else p.get(key, [])
            q[key] = [mapping[i] for i in parse_indices(value, len(a['symbols']))]
        if 'anchor' in p:
            q['anchor'] = mapping[int(p['anchor'])]
        for key in ('reaction_edits', 'generation_edits', 'bond_overrides'):
            if p.get(key):
                q[key] = {
                    k: [[mapping[int(i)] for i in bond] for bond in pairs] for k,
                    pairs in p[key].items()
                }
        q['atom_mapping_source'] = source_label
        q['mapping_alternative'] = index
        q['mapping_reviewed'] = p.get('mapping_confirmed') is True
        out.append({
            'params': q,
            'mapping': mapping,
            'source': source_label,
            'manual_reviewed': q['mapping_reviewed']
        })
    return out
