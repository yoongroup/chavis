from __future__ import annotations
import copy
import itertools
import math
from .geometry import validate, parse_indices, add, sub, mul, dot, cross, norm, unit, rotate, basis, RADII
from .collision import GeometryError, dual as _dual, translations as _translations, minimum_vector, collisions as _collisions


def _number(p, key, default, lower=None, upper=None):
    v = float(p.get(key, default))
    if not math.isfinite(v) or (lower is not None and v < lower) or (upper is not None and v > upper):
        raise GeometryError(f'{key}: out of range or not a finite number ({v}).')
    return v


def _flat(v, n):
    return sub(v, mul(n, dot(v, n)))


def _canonical(point, s):
    g0, g1, _ = _dual(s)
    q = point[:]
    for j, g in enumerate((g0, g1)):
        if s['pbc'][j]:
            q = sub(q, mul(s['cell'][j], math.floor(dot(point, g) + 1e-09)))
    return q


def _unique_site(sites, point, kind, members, s, tolerance=0.025):
    p = _canonical(point, s)
    for site in sites:
        if site['kind'] == kind and norm(minimum_vector(sub(p, site['position']), s)) < tolerance:
            return
    sites.append({'kind': kind, 'position': p, 'surface_atoms': sorted(set(members))})


def _facet(s, params):
    saved = (s.get('meta') or {}).get('surface_profile') or {}
    declared = params.get('facet')
    if declared and saved and (declared != saved.get('facet')):
        raise GeometryError('Requested facet differs from saved surface profile.')
    facet = declared or saved.get('facet') or 'fcc111'
    if facet not in ('fcc111', 'fcc100'):
        raise GeometryError('Unsupported site facet: ' + str(facet))
    if facet == 'fcc100' and (not saved or saved.get('facet') != 'fcc100'):
        raise GeometryError('Ni(100) site generation requires a saved fcc100 surface profile.')
    return facet


def enumerate_square_sites(parent, adsorbate, params=None):
    p = params or {}
    s = validate(parent)
    n = len(s['symbols'])
    if _facet(s, p) != 'fcc100':
        raise GeometryError('Square site generator requires fcc100 profile.')
    if not all(s['pbc'][:2]) or s['pbc'][2]:
        raise GeometryError('fcc100 sites require two in-plane periodic axes only.')
    slab = [i for i in range(n) if i not in set(adsorbate)]
    if not slab:
        raise GeometryError('No substrate atoms remain.')
    normal = _dual(s)[2]
    tol = _number(p, 'layer_tol', 0.35, 0.02, 1.0)
    z = {i: dot(s['positions'][i], normal) for i in slab}
    layers = []
    for i in sorted(slab, key=z.get, reverse=True):
        if not layers or layers[-1][0] - z[i] > tol:
            layers.append([z[i], [i]])
        else:
            layers[-1][1].append(i)
    top = parse_indices(p.get('surface_indices', ''), n) or layers[0][1]
    if not top or any((i not in slab for i in top)):
        raise GeometryError('Surface selection is empty or includes adsorbate atoms.')
    if max((z[i] for i in top)) - min((z[i] for i in top)) > tol:
        raise GeometryError('fcc100 site generator requires a flat exposed top layer.')
    span = max(norm(s['cell'][0]), norm(s['cell'][1]))
    distances = []
    for i in top:
        for j in top:
            d0 = sub(s['positions'][i], s['positions'][j])
            distances.extend((
                norm(sub(d0, t)) for _,
                t in _translations(d0, s, min(span, 6.0)) if norm(sub(d0, t)) > 0.05
            ))
    if not distances:
        raise GeometryError('Cannot determine top-layer spacing.')
    nn = min(distances)
    cutoff = nn * 1.13
    if nn < 1 or nn > 4:
        raise GeometryError('Unexpected fcc100 surface spacing.')
    sites = []
    for i in top:
        ri = s['positions'][i]
        neighbors = []
        for j in top:
            for shift, t in _translations(sub(ri, s['positions'][j]), s, cutoff):
                rj = add(s['positions'][j], t)
                v = sub(rj, ri)
                d = norm(v)
                if nn * 0.87 < d < cutoff:
                    neighbors.append((j, shift, rj, v))
        if len(neighbors) != 4:
            raise GeometryError(f'Atom {i} has {len(neighbors)} in-plane nearest neighbours (expected 4 for fcc100).')
        _unique_site(sites, ri, 'top', [i], s)
        for j, _, rj, _ in neighbors:
            _unique_site(sites, mul(add(ri, rj), 0.5), 'bridge', [i, j], s)
        for (j, _, rj, vj), (k, _, rk, vk) in itertools.combinations(neighbors, 2):
            if abs(dot(vj, vk)) > 0.13 * nn * nn:
                continue
            fourth = add(add(ri, vj), vk)
            matches = [(
                q,
                norm(minimum_vector(sub(fourth, s['positions'][q]), s))
            ) for q in top]
            q, residual = min(matches, key=lambda item: item[1])
            if residual > 0.13 * nn:
                continue
            center = mul(add(add(ri, rj), add(rk, fourth)), 0.25)
            _unique_site(sites, center, 'fourfold', [i, j, k, q], s)
    sites.sort(key=lambda v: (v['kind'], *[round(x, 5) for x in v['position']]))
    for j, site in enumerate(sites):
        site['site_id'] = f"{site['kind']}_{j + 1:03d}"
    return (
        sites,
        {
            'site_count': len(sites),
            'top_layer_indices': top,
            'layer_count': len(layers),
            'nearest_neighbour_A': nn,
            'normal': normal,
            'layer_tolerance_A': tol,
            'assumption': 'flat near-ideal fcc(100) square terrace; geometric site labels only'
        }
    )


def enumerate_terrace_sites(parent, adsorbate, params=None):
    p = params or {}
    s = validate(parent)
    n = len(s['symbols'])
    if _facet(s, p) == 'fcc100':
        return enumerate_square_sites(s, adsorbate, p)
    _dual(s)
    if not all(s['pbc'][:2]):
        raise GeometryError('Surface-site enumeration requires in-plane PBC in both cell[0] and cell[1].')
    slab = [i for i in range(n) if i not in set(adsorbate)]
    if not slab:
        raise GeometryError('No slab atoms remain. Check Adsorbate indices.')
    normal = _dual(s)[2]
    tol = _number(p, 'layer_tol', 0.35, 0.02, 1.0)
    z = {i: dot(s['positions'][i], normal) for i in slab}
    ordered = sorted(slab, key=z.get, reverse=True)
    layers = []
    for i in ordered:
        if not layers or layers[-1][0] - z[i] > tol:
            layers.append([z[i], [i]])
        else:
            layers[-1][1].append(i)
    manual = parse_indices(p.get('surface_indices', ''), n)
    top = manual or layers[0][1]
    if any((i not in slab for i in top)):
        raise GeometryError('Surface indices include an adsorbate atom.')
    if len(top) < 1:
        raise GeometryError('No surface atoms found.')
    if max((z[i] for i in top)) - min((z[i] for i in top)) > tol:
        raise GeometryError('Selected surface atoms are not a flat layer. Use a custom generator for steps/reconstructions.')
    span = max(norm(s['cell'][0]), norm(s['cell'][1]))
    distances = []
    for i in top:
        for j in top:
            d0 = sub(s['positions'][i], s['positions'][j])
            for (u, v), t in _translations(d0, s, min(span, 6.0)):
                d = norm(sub(d0, t))
                if d > 0.05:
                    distances.append(d)
    if not distances:
        raise GeometryError('Cannot determine nearest-neighbour distance in the top layer.')
    nn = min(distances)
    cutoff = nn * 1.13
    if nn < 1 or nn > 4:
        raise GeometryError(f'Unexpected surface spacing {nn:.3f} A. Check units and surface selection.')
    sites = []
    for i in top:
        ri = s['positions'][i]
        neighbors = []
        for j in top:
            for (u, v), t in _translations(sub(ri, s['positions'][j]), s, cutoff):
                rj = add(s['positions'][j], t)
                d = norm(sub(rj, ri))
                if nn * 0.87 < d < cutoff:
                    neighbors.append((j, (u, v), rj))
        if len(neighbors) != 6:
            raise GeometryError(f'Atom {i} has {len(neighbors)} in-plane nearest neighbours (expected 6). This engine supports flat fcc(111) terraces; check layer_tol or use your generator.')
        _unique_site(sites, ri, 'top', [i], s)
        for j, shift, rj in neighbors:
            _unique_site(sites, mul(add(ri, rj), 0.5), 'bridge', [i, j], s)
        for (j, sh1, rj), (k, sh2, rk) in itertools.combinations(neighbors, 2):
            if not nn * 0.87 < norm(sub(rj, rk)) < cutoff:
                continue
            center = mul(add(add(ri, rj), rk), 1 / 3)
            under = []
            for _, indices in layers[1:3]:
                under.append(min(
                    (norm(_flat(
                        minimum_vector(sub(center, s['positions'][q]), s),
                        normal
                    )) for q in indices),
                    default=999
                ))
            eps = nn * 0.18
            if under and under[0] < eps:
                kind = 'hcp'
            elif len(under) > 1 and under[1] < eps:
                kind = 'fcc'
            else:
                kind = 'hollow_unknown'
            _unique_site(sites, center, kind, [i, j, k], s)
    sites.sort(key=lambda v: (v['kind'], *[round(x, 5) for x in v['position']]))
    for j, site in enumerate(sites):
        site['site_id'] = f"{site['kind']}_{j + 1:03d}"
    summary = {
        'site_count': len(sites),
        'top_layer_indices': top,
        'layer_count': len(layers),
        'nearest_neighbour_A': nn,
        'normal': normal,
        'layer_tolerance_A': tol,
        'assumption': 'flat near-ideal fcc(111) terrace; geometric site labels only'
    }
    return (sites, summary)


def infer_graph(s, ads, scale=1.2):
    from .connectivity import connectivity
    return connectivity(s, ads, scale)[0]


def component(g, start):
    todo = [start]
    seen = set()
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        todo.extend(g[i] - seen)
    return seen


def suggest_bonds(parent, params):
    s = validate(parent)
    ads = parse_indices(
        params.get('adsorbate', s.get('adsorbate_indices', [])),
        len(s['symbols'])
    )
    if not ads:
        raise GeometryError('Specify Adsorbate indices before bond inference.')
    g = infer_graph(s, ads, float(params.get('bond_scale', 1.2)))
    out = []
    for i in ads:
        for j in sorted(g[i]):
            if j <= i:
                continue
            h = {k: set(v) for k, v in g.items()}
            h[i].remove(j)
            h[j].remove(i)
            side_i = component(h, i)
            side_j = component(h, j)
            if j in side_i:
                continue
            a, b = (i, j) if len(side_j) <= len(side_i) else (j, i)
            moving = sorted(component(h, b))
            moving.remove(b)
            moving.insert(0, b)
            out.append({
                'anchor': a,
                'moving': moving,
                'cut_bond': [i, j],
                'label': f"{s['symbols'][i]}{i}–{s['symbols'][j]}{j} → move " + ','.join(map(str, moving))
            })
    return out


def _unwrap_adsorbates(s, ads, g):
    out = copy.deepcopy(s)
    seen = set()
    changed = []
    for root in ads:
        if root in seen:
            continue
        todo = [root]
        seen.add(root)
        while todo:
            i = todo.pop()
            for j in g[i]:
                target = add(
                    out['positions'][i],
                    minimum_vector(sub(s['positions'][j], s['positions'][i]), s)
                )
                if j in seen:
                    if norm(sub(target, out['positions'][j])) > 0.05:
                        raise GeometryError('Adsorbate graph winds through PBC. This engine expects finite molecules, not polymers.')
                    continue
                if norm(sub(target, out['positions'][j])) > 1e-07:
                    changed.append(j)
                out['positions'][j] = target
                seen.add(j)
                todo.append(j)
    return (out, changed)


''
''


def generate_surface(parent, params):
    p = copy.deepcopy(params)
    original = validate(parent)
    s = copy.deepcopy(original)
    n = len(s['symbols'])
    _dual(s)
    mode = p.get('mode', 'cleavage')
    if mode not in ('cleavage', 'adsorption', 'desorption', 'association', 'transfer'):
        raise GeometryError('Supported modes: cleavage, adsorption, desorption, association, transfer.')
    ads = parse_indices(p.get('adsorbate', s.get('adsorbate_indices', [])), n)
    if not ads:
        raise GeometryError('Adsorbate indices are required; slab atoms must not be included.')
    anchor = int(p.get('anchor', ads[0]))
    moving = parse_indices(p.get('moving', ''), n)
    if not 0 <= anchor < n or anchor not in ads or (not moving) or (not set(moving) <= set(ads)):
        raise GeometryError('Anchor/moving atoms must be valid adsorbate indices.')
    lead = moving[0]
    from .connectivity import connectivity
    g, graph_backend = connectivity(
        s,
        ads,
        _number(p, 'bond_scale', 1.2, 0.9, 1.6),
        p.get('graph_backend', 'auto'),
        p.get('bond_overrides')
    )
    original_graph = {i: set(v) for i, v in g.items()}
    edits = copy.deepcopy(p.get('reaction_edits') or {'break': [], 'form': []})
    for pair in edits.get('break', []):
        if len(pair) != 2 or any((int(i) not in g for i in pair)):
            raise GeometryError('Break edits must reference adsorbate atoms.')
        i, j = map(int, pair)
        if j not in g[i]:
            raise GeometryError('Break edit is not a current inferred bond. Clear stale edits or correct connectivity.')
    if edits.get('form') and mode not in ('association', 'transfer'):
        raise GeometryError('Form-bond edits need Association/Transfer or a custom generator.')
    if mode in ('association', 'transfer') and edits.get('form'):
        if len(edits['form']) != 1 or set(edits['form'][0]) != {anchor, lead}:
            raise GeometryError('Docking supports one forming bond matching anchor and moving[0].')
    if mode == 'association' and edits.get('break'):
        raise GeometryError('Use Transfer mode for a break-plus-form edit.')
    unwrapped = []
    if p.get('unwrap_adsorbate', True):
        s, unwrapped = _unwrap_adsorbates(s, ads, g)
    if mode == 'cleavage':
        if anchor == lead or lead not in g[anchor]:
            raise GeometryError('The chosen anchor–moving[0] bond was not inferred. Check indices or bond_scale.')
        cut = {i: set(v) for i, v in g.items()}
        cuts = edits.get('break') or [[anchor, lead]]
        if {anchor, lead} not in [set(x) for x in cuts]:
            raise GeometryError('Selected cleavage does not match reaction edits.')
        for i, j in cuts:
            cut[i].discard(j)
            cut[j].discard(i)
        edits['break'] = cuts
        auto = component(cut, lead)
        if anchor in auto:
            raise GeometryError('Deleting this single bond does not split the molecular graph (ring/multiple connection).')
        if p.get('fragment_policy', 'auto') == 'auto':
            moving = [lead] + sorted(auto - {lead})
        elif set(moving) != auto:
            raise GeometryError('Manual moving indices cut additional inferred bonds. Use the full fragment or Auto fragment.')
    elif mode in ('adsorption', 'desorption'):
        auto = component(g, lead)
        if p.get('fragment_policy', 'auto') == 'auto':
            moving = [lead] + sorted(auto - {lead})
        elif set(moving) != auto:
            raise GeometryError('Move the entire molecular component for adsorption/desorption.')
    else:
        if mode == 'transfer':
            cuts = edits.get('break', [])
            forms = edits.get('form', [])
            if len(cuts) != 1 or len(forms) != 1 or lead not in cuts[0] or (set(forms[0]) != {anchor, lead}):
                raise GeometryError('Transfer supports one broken and one formed bond sharing moving[0].')
            i, j = cuts[0]
            g[i].discard(j)
            g[j].discard(i)
        if anchor in component(g, lead):
            raise GeometryError('Association needs two separate inferred adsorbate components. Select atoms on different fragments.')
        if p.get('fragment_policy', 'auto') == 'auto':
            moving = [lead] + sorted(component(g, lead) - {lead})
    if mode == 'cleavage':
        for i, j in edits.get('break', []):
            if (i in moving) == (j in moving):
                raise GeometryError('A requested break is inside a retained/moving component and would not change. Use a custom multi-fragment generator.')
    fixed = set(s.get('fixed_indices', []))
    if fixed.intersection(moving):
        raise GeometryError('Moving fragment includes fixed atoms. Review constraints first.')
    x, y, normal = basis(s)
    rotation = _number(p, 'rotation', 0)
    origin = s['positions'][anchor][:]
    if rotation:
        rotating = component(original_graph, anchor) if mode == 'cleavage' else set(moving)
        if fixed.intersection(rotating):
            raise GeometryError('Rotation includes fixed atoms. Review constraints first.')
        for i in rotating:
            s['positions'][i] = add(
                origin,
                rotate(sub(s['positions'][i], origin), normal, math.radians(rotation))
            )
    s.pop('energy', None)
    s['adsorbate_indices'] = ads
    s['meta'] = {
        **s.get('meta', {}),
        'adsorbate_indices': ads,
        'periodic_unwrapped_atom_indices': unwrapped
    }
    origin = s['positions'][anchor][:]
    leadpos = s['positions'][lead][:]
    v = sub(leadpos, origin)
    axis = unit(v) if norm(v) > 1e-09 else x
    planar = _flat(axis, normal)
    vertical = norm(planar) < 1e-07
    half = _number(p, 'half_angle', 30, 0, 180)
    domain = p.get('domain', 'sector')
    if domain not in ('sector', 'cone3d'):
        raise GeometryError('Choose sector or cone3d domain.')
    if domain == 'sector' and vertical and (mode in ('cleavage', 'association', 'transfer')):
        if half < 180:
            raise GeometryError('Bond projection on the surface is zero. Use a 3D cone or set half-angle=180 for all azimuths.')
        planar = x
    flat = unit(planar) if norm(planar) > 1e-08 else x
    rmin = _number(p, 'rmin', 1.8, 0 if mode == 'adsorption' else 0.05, 20)
    rmax = _number(p, 'rmax', 3, 0.05, 20)
    if rmax < rmin:
        raise GeometryError('r max must be >= r min.')
    height = _number(p, 'height', 1.0, 0.05, 10)
    scale = _number(p, 'collision_scale', 0.7, 0.1, 1.2)
    height_mode = p.get('height_mode', 'fixed')
    if height_mode not in ('fixed', 'drop'):
        raise GeometryError('Height mode must be fixed or drop.')
    contact_scale = _number(p, 'contact_scale', 1.0) if height_mode == 'drop' else None
    if height_mode == 'drop':
        if mode != 'cleavage':
            raise GeometryError('Drop-contact placement is restricted to cleavage.')
        if contact_scale <= 0:
            raise GeometryError('Surface contact scale must exceed 0.')
        if scale > contact_scale:
            raise GeometryError('Collision scale cannot exceed surface contact scale.')
    na = int(p.get('angular_count', 5))
    nr = int(p.get('radial_count', 3))
    if not 1 <= na <= 36 or not 1 <= nr <= 20:
        raise GeometryError('Counts must be angular 1..36, radial 1..20.')
    targets = []
    summary = {}
    if mode in ('cleavage', 'adsorption'):
        from chavis.inputs.library_sites import enumerate_sites
        sites, summary = enumerate_sites(s, ads, p)
        types = p.get('site_types', 'top,bridge,fcc,hcp,hollow_unknown,fourfold')
        if isinstance(types, str):
            types = [t.strip() for t in types.split(',') if t.strip()]
        if not types or set(types) - {
            'top',
            'bridge',
            'fcc',
            'hcp',
            'hollow_unknown',
            'fourfold'
        }:
            raise GeometryError('Select top, bridge, fcc, hcp, hollow_unknown, and/or fourfold.')
        envelope = rmax + 0.75
        for site in sites:
            if site['kind'] not in types:
                continue
            base = add(site['position'], mul(normal, height))
            v0 = _flat(sub(origin, base), normal)
            for ij, t in _translations(v0, s, envelope):
                target = add(base, t)
                disp = sub(target, origin)
                planar_dist = norm(_flat(disp, normal))
                if planar_dist > envelope + 1e-07:
                    continue
                contact = None
                if height_mode == 'drop':
                    from chavis.inputs.contact_placement import contact_target
                    try:
                        target, contact = contact_target(
                            s,
                            moving,
                            ads,
                            add(site['position'], t),
                            normal,
                            contact_scale
                        )
                    except ValueError as error:
                        raise GeometryError(f"{site['site_id']} / image {ij}: {error}") from error
                    disp = sub(target, origin)
                offset = 0 if planar_dist < 1e-08 else math.degrees(math.acos(max(
                    -1,
                    min(1, dot(unit(_flat(disp, normal)), flat))
                )))
                polar = 0 if norm(disp) < 1e-08 else math.degrees(math.acos(max(-1, min(1, dot(unit(disp), axis)))))
                distance = norm(disp) if domain == 'cone3d' else planar_dist
                angle = polar if domain == 'cone3d' else offset
                reasons = []
                if distance < rmin - 1e-07 or distance > rmax + 1e-07:
                    reasons.append('radius')
                if mode == 'cleavage' and angle > half + 1e-07:
                    reasons.append('angle')
                targets.append((
                    target,
                    {
                        'site': site['kind'],
                        'site_id': site['site_id'],
                        'cell_shift': list(ij),
                        'surface_atoms': site['surface_atoms'],
                        'distance_A': distance,
                        'planar_r_A': planar_dist,
                        'polar_deg': polar,
                        'azimuth_offset_deg': offset,
                        'height_A': height
                    },
                    reasons
                ))
                if contact is not None:
                    targets[-1][1].update(
                        height_mode='drop',
                        height_A=contact['height_from_site_A'],
                        contact=contact
                    )
    elif mode == 'desorption':
        for k in range(nr):
            d = rmin if nr == 1 else rmin + (rmax - rmin) * k / (nr - 1)
            targets.append((
                add(leadpos, mul(normal, d)),
                {'displacement_A': d, 'site': 'normal'},
                []
            ))
    else:
        u = unit(cross(axis, normal)) if norm(cross(axis, normal)) > 1e-08 else x
        w = unit(cross(axis, u))
        for kr in range(nr):
            r = rmin if nr == 1 else rmin + (rmax - rmin) * kr / (nr - 1)
            directions = [(axis, 0.0)]
            if half > 0:
                for k in range(na):
                    phi = 2 * math.pi * k / na
                    theta = math.radians(half)
                    directions.append((
                        add(
                            mul(axis, math.cos(theta)),
                            mul(
                                add(mul(u, math.cos(phi)), mul(w, math.sin(phi))),
                                math.sin(theta)
                            )
                        ),
                        half
                    ))
            for d, theta in directions:
                targets.append((
                    add(origin, mul(d, r)),
                    {
                        'binding_distance_A': r,
                        'polar_deg': theta,
                        'site': 'association_guess'
                    },
                    []
                ))
    if len(targets) > 300:
        raise GeometryError(f'{len(targets)} candidates exceed preview limit 300. Reduce radius/sites/counts; none were silently truncated.')
    if not targets:
        raise GeometryError('No sites in this search neighbourhood. Check atom indices, height, radii and surface indices.')
    out = []
    for index, (target, detail, reasons) in enumerate(targets):
        q = copy.deepcopy(s)
        d = sub(target, leadpos)
        for i in moving:
            q['positions'][i] = add(s['positions'][i], d)
        collisions = _collisions(q, moving, scale)
        if collisions:
            reasons.append('collision')
        if mode == 'cleavage':
            bondlen = norm(minimum_vector(
                sub(q['positions'][lead], q['positions'][anchor]),
                q
            ))
            if q['symbols'][lead] == q['symbols'][anchor] == 'H':
                from chavis.judgment.policy import bond_threshold
                cutoff = bond_threshold('H', 'H')
            else:
                cutoff = float(p.get('bond_scale', 1.2)) * (RADII.get(q['symbols'][lead], 0.8) + RADII.get(q['symbols'][anchor], 0.8))
            if bondlen < cutoff:
                reasons.append('target_bond_still_close')
        if original['pbc'][2]:
            cell_z = abs(dot(original['cell'][2], normal))
            slab = [i for i in range(n) if i not in ads]
            top = max((dot(q['positions'][i], normal) for i in slab))
            bottom = min((dot(q['positions'][i], normal) for i in slab))
            for i in moving:
                zi = dot(q['positions'][i], normal)
                if zi > bottom + cell_z - 1.5 or zi < top - cell_z + 1.5:
                    reasons.append('vacuum_image_too_close')
                    break
        detail.update({
            'mode': mode,
            'anchor': anchor,
            'moving': moving,
            'rotation_deg': rotation,
            'domain': domain,
            'collision_pairs': collisions[:20],
            'filters': list(dict.fromkeys(reasons)),
            'surface_summary': summary,
            'periodic_unwrapped_atom_indices': unwrapped,
            'reaction_edits': edits,
            'graph_backend': graph_backend
        })
        from .generation_diagnostics import record_checks
        detail['validation'] = record_checks(q, detail, p, reasons, collisions, normal)
        q['meta'].update(
            generated=True,
            unrelaxed=True,
            generator='local_terrace_sites_v0_2',
            parameters=detail
        )
        guide_domain = 'cone3d' if mode in ('association', 'transfer') else domain
        guide = {
            'origin': origin,
            'axis': axis if guide_domain == 'cone3d' else flat,
            'normal': normal,
            'x': x,
            'half_angle': half,
            'rmin': rmin,
            'rmax': rmax,
            'domain': guide_domain,
            'target': target,
            'moving': moving,
            'anchor': anchor,
            'geometry_only': True
        }
        if mode == 'desorption':
            guide = None
        out.append({
            'name': f"{mode}_{index + 1:03d}_{detail['site']}.extxyz",
            'structure': q,
            'input_parent': copy.deepcopy(s),
            'accepted': not reasons,
            'reason': ', '.join(dict.fromkeys(reasons)),
            'detail': detail,
            'guides': guide
        })
    print(f"Surface endpoint generator: {mode}, {len(out)} candidates; {sum((c['accepted'] for c in out))} retained.")
    print('Geometric guesses only: no relaxation, energy, or TS validation. Deduplication is a separate review step.')
    if summary:
        print(f"Sites from input slab: {summary['site_count']}; layers: {summary['layer_count']}; top indices: {summary['top_layer_indices']}")
    if unwrapped:
        print('PBC-unwrapped atom indices (same periodic geometry):', unwrapped)
    return out
