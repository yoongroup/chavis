from __future__ import annotations
import copy, math
from collections import Counter
from geometry import validate, parse_indices, sub, norm, dot, cross, mul, add, unit, basis, RADII
from library_sites import layer_info


def displacement(a, b, s):
    from chavis.search.connectivity import displacement as shared_displacement
    return shared_displacement(a, b, s)


''


def connectivity(s, ads, scale=1.2, backend='auto', overrides=None):
    from chavis.search.connectivity import connectivity as shared_connectivity
    return shared_connectivity(
        s,
        ads,
        scale=scale,
        backend=backend,
        overrides=overrides
    )


def check_pair(pair, g):
    from chavis.search.connectivity import check_pair as shared_check_pair
    return shared_check_pair(pair, g)


def component(g, start):
    if start not in g:
        raise ValueError('Selected atom is not in the adsorbate graph.')
    todo = [start]
    seen = set()
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        todo.extend(g[i] - seen)
    return seen


def components(g):
    out = []
    seen = set()
    for i in sorted(g):
        if i not in seen:
            group = component(g, i)
            seen |= group
            out.append(sorted(group))
    return out


def formula(s, ids):
    c = Counter((s['symbols'][i] for i in ids))
    order = sorted(c, key=lambda e: (0 if e == 'C' else 1 if e == 'H' else 2, e))
    return ''.join((e + (str(c[e]) if c[e] > 1 else '') for e in order))


def analyse(parent, params):
    s = validate(parent)
    ads = parse_indices(
        params.get('adsorbate', s.get('adsorbate_indices', [])),
        len(s['symbols'])
    )
    g, backend = connectivity(
        s,
        ads,
        float(params.get('bond_scale', 1.2)),
        params.get('graph_backend', 'auto'),
        params.get('bond_overrides')
    )
    return {
        **layer_info(s, ads, float(params.get('layer_tol', 0.35))),
        'graph_backend': backend,
        'bonds': [{
            'indices': [i, j],
            'distance_A': norm(displacement(s['positions'][i], s['positions'][j], s))
        } for i in g for j in sorted(g[i]) if i < j],
        'components': [{'indices': x, 'formula': formula(s, x)} for x in components(g)],
        'warning': 'Geometry-only connectivity; verify adsorbate membership and ambiguous bonds.'
    }


''


def reaction_selection(parent, params, action, indices, edits=None):
    s = validate(parent)
    ads = parse_indices(
        params.get('adsorbate', s.get('adsorbate_indices', [])),
        len(s['symbols'])
    )
    g, backend = connectivity(
        s,
        ads,
        float(params.get('bond_scale', 1.2)),
        params.get('graph_backend', 'auto'),
        params.get('bond_overrides')
    )
    ids = parse_indices(indices, len(s['symbols']))
    if not set(ids) <= set(ads):
        raise ValueError('Reaction selection is limited to explicit adsorbate atoms. Edit adsorbate selection first.')
    edits = copy.deepcopy(edits or {'break': [], 'form': []})
    cuts = edits.setdefault('break', [])
    forms = edits.setdefault('form', [])
    if action == 'break':
        i, j = check_pair(ids, g)
        if j not in g[i]:
            raise ValueError('Selected pair is not an inferred bond. Review cutoff or override connectivity.')
        if {i, j} not in [set(x) for x in cuts]:
            cuts.append([i, j])
    elif action == 'form':
        i, j = check_pair(ids, g)
        if j in g[i]:
            raise ValueError('Selected pair is already connected.')
        if {i, j} not in [set(x) for x in forms]:
            forms.append([i, j])
    elif action == 'transfer':
        if len(ids) != 3:
            raise ValueError('Select donor, transferred atom, acceptor in that order.')
        donor, transfer, acceptor = ids
        if transfer not in g[donor]:
            raise ValueError('Donor and transferred atom are not connected.')
        if acceptor in g[transfer]:
            raise ValueError('Transferred atom is already bonded to the acceptor.')
        cuts = [[donor, transfer]]
        forms = [[transfer, acceptor]]
        edits = {'break': cuts, 'form': forms}
    elif action not in ('component', 'preview'):
        raise ValueError('Unknown selection action.')
    cutgraph = {i: set(v) for i, v in g.items()}
    for pair in cuts:
        i, j = check_pair(pair, g)
        if j not in g[i]:
            raise ValueError(f'Break bond {i}–{j} is not in the inferred graph.')
        cutgraph[i].discard(j)
        cutgraph[j].discard(i)
    mode = params.get('mode', 'cleavage')
    anchor = None
    moving = []
    warning = ''
    if action == 'component':
        if len(ids) != 1:
            raise ValueError('Click one atom to select its connected fragment.')
        moving = [ids[0]] + sorted(component(cutgraph, ids[0]) - {ids[0]})
    elif cuts and forms:
        if len(cuts) != 1 or len(forms) != 1:
            warning = 'Multiple bond-forming edits are recorded; automatic transfer placement supports one break + one form only.'
        else:
            common = set(cuts[0]) & set(forms[0])
            if len(common) == 1:
                lead = next(iter(common))
                anchor = next((i for i in forms[0] if i != lead))
                moving = [lead] + sorted(component(cutgraph, lead) - {lead})
                mode = 'transfer'
                if anchor in moving:
                    warning = 'Transfer needs a separate accepting fragment after the cut. Use a custom generator.'
            else:
                warning = 'These edits need a custom concerted geometry generator.'
    elif cuts:
        i, j = cuts[-1]
        a = component(cutgraph, i)
        b = component(cutgraph, j)
        if j in a:
            warning = 'Cut does not separate a ring. Add another cut or use a custom ring-opening generator.'
        else:
            anchor, lead = (i, j) if len(b) <= len(a) else (j, i)
            moving = [lead] + sorted(component(cutgraph, lead) - {lead})
            mode = 'cleavage'
    elif forms:
        if len(forms) > 1:
            warning = 'Automatic docking supports one forming bond; use a custom generator for several.'
        i, j = forms[-1]
        if j in component(cutgraph, i):
            warning = 'Both atoms remain in the same fragment. Ring closure requires a custom generator.'
        else:
            anchor, lead = (i, j) if len(component(cutgraph, j)) <= len(component(cutgraph, i)) else (j, i)
            moving = [lead] + sorted(component(cutgraph, lead) - {lead})
            mode = 'association'
    groups = components(cutgraph)
    return {
        'edits': edits,
        'mode': mode,
        'anchor': anchor,
        'moving': moving,
        'fragments': [{'indices': x, 'formula': formula(s, x)} for x in groups],
        'backend': backend,
        'warning': warning,
        'can_apply': bool(moving) and (not warning)
    }


''


def measure(parent, indices, kind):
    s = validate(parent)
    ids = parse_indices(indices, len(s['symbols']))
    n = basis(s)[2]
    req = {'distance': 2, 'angle': 3, 'dihedral': 4, 'normal': 2}
    if kind not in req or len(ids) != req[kind]:
        raise ValueError('Select ' + str(req.get(kind, '?')) + ' atoms for ' + kind)
    p = [s['positions'][i] for i in ids]
    d = lambda a, b: displacement(a, b, s)
    acos = lambda x: math.degrees(math.acos(max(-1.0, min(1.0, x))))
    if kind == 'distance':
        value = norm(d(p[0], p[1]))
        unit_name = 'Å'
    elif kind == 'normal':
        value = acos(dot(unit(d(p[0], p[1])), n))
        unit_name = '°'
    elif kind == 'angle':
        value = acos(dot(unit(d(p[1], p[0])), unit(d(p[1], p[2]))))
        unit_name = '°'
    else:
        b0 = mul(d(p[0], p[1]), -1)
        b1 = unit(d(p[1], p[2]))
        b2 = d(p[2], p[3])
        v = sub(b0, mul(b1, dot(b0, b1)))
        w = sub(b2, mul(b1, dot(b2, b1)))
        if norm(v) < 1e-10 or norm(w) < 1e-10:
            raise ValueError('Dihedral is undefined for collinear atoms.')
        value = math.degrees(math.atan2(dot(cross(b1, v), w), dot(v, w))) % 360
        unit_name = '°'
    drawn = [p[0][:]]
    for a, b in zip(p, p[1:]):
        drawn.append(add(drawn[-1], d(a, b)))
    return {
        'kind': kind,
        'indices': ids,
        'value': value,
        'unit': unit_name,
        'points': drawn,
        'pbc': 'in-plane MIC; normal direction not wrapped'
    }


def map_recipe(source, target, params):
    a = validate(source)
    b = validate(target)
    p = copy.deepcopy(params)
    byid = {v: i for i, v in enumerate(b['atom_ids'])}
    mapping = {}
    for i, id_ in enumerate(a['atom_ids']):
        j = byid.get(id_)
        if j is None or a['symbols'][i] != b['symbols'][j]:
            raise ValueError('Atom IDs/elements differ. Choose the bond in this mother separately.')
        mapping[i] = j
    if len(a['symbols']) != len(b['symbols']):
        raise ValueError('Different compositions: this recipe requires explicit reselection.')
    ads = parse_indices(
        p.get('adsorbate', a.get('adsorbate_indices', [])),
        len(a['symbols'])
    )
    ga, _ = connectivity(a, ads, float(p.get('bond_scale', 1.2)))
    gb, _ = connectivity(b, [mapping[i] for i in ads], float(p.get('bond_scale', 1.2)))
    if any(({mapping[j] for j in ga[i]} != gb[mapping[i]] for i in ads)):
        raise ValueError('Adsorbate connectivity differs; recipe was not applied.')
    for key in ('adsorbate', 'moving', 'surface_indices'):
        p[key] = [mapping[i] for i in parse_indices(p.get(key, []), len(a['symbols']))]
    p['anchor'] = mapping[int(p['anchor'])]
    for field in ('reaction_edits', 'bond_overrides'):
        if p.get(field):
            p[field] = {
                k: [[mapping[i] for i in pair] for pair in pairs] for k,
                pairs in p[field].items()
            }
    return p
