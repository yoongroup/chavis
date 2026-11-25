from __future__ import annotations
from chavis.search.geometry import validate, parse_indices, basis, dot, norm, sub, mul


def capabilities():
    import importlib.util
    return {name: importlib.util.find_spec(name) is not None for name in (
        'ase',
        'numpy',
        'scipy',
        'spglib',
        'pymatgen',
        'networkx'
    )}


def layer_info(parent, adsorbate, tol=0.35):
    s = validate(parent)
    if not 0.01 <= float(tol) <= 2:
        raise ValueError('Layer tolerance must be 0.01..2 A.')
    ads = set(parse_indices(adsorbate, len(s['symbols'])))
    n = basis(s)[2]
    slab = [i for i in range(len(s['symbols'])) if i not in ads]
    heights = {i: dot(s['positions'][i], n) for i in slab}
    layers = []
    for i in sorted(slab, key=heights.get, reverse=True):
        if not layers or layers[-1]['top_A'] - heights[i] > tol:
            layers.append({
                'layer': len(layers) + 1,
                'top_A': heights[i],
                'indices': []
            })
        layers[-1]['indices'].append(i)
    for layer in layers:
        layer['bottom_A'] = min((heights[i] for i in layer['indices']))
        layer['symbols'] = sorted({s['symbols'][i] for i in layer['indices']})
    return {
        'layers': layers,
        'normal': n,
        'adsorbate_indices': sorted(ads),
        'slab_indices': slab
    }


def enumerate_sites(parent, adsorbate, params=None):
    from chavis.search.surface_sites import enumerate_terrace_sites
    p = params or {}
    backend = p.get('site_backend', 'existing_terrace')
    if backend == 'existing_terrace':
        sites, summary = enumerate_terrace_sites(parent, adsorbate, p)
        return (sites, {**summary, 'backend': 'existing_terrace_v0_2 (preserved)'})
    if backend != 'pymatgen':
        raise ValueError('Unknown site backend: ' + str(backend))
    return pymatgen_sites(parent, adsorbate, p)


def pymatgen_sites(parent, adsorbate, params=None):
    p = params or {}
    s = validate(parent)
    profile = (s.get('meta') or {}).get('surface_profile') or {}
    if profile.get('facet') == 'fcc100':
        raise ValueError('pymatgen site backend is not validated for fcc100; select existing_terrace.')
    try:
        import numpy as np
        from pymatgen.core import Structure
        try:
            from pymatgen.core.adsorption import AdsorbateSiteFinder
        except ImportError:
            from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    except ImportError as e:
        raise ValueError('pymatgen site backend requires INSTALL_PYMATGEN.command (or .bat). Existing terrace backend is still available; no silent fallback.') from e
    from chavis.search.collision import dual as _dual, minimum_vector
    from chavis.search.surface_sites import _unique_site
    normal = _dual(s)[2]
    ads = parse_indices(adsorbate, len(s['symbols']))
    if not all(s['pbc'][:2]):
        raise ValueError('Site generation requires in-plane PBC.')
    if abs(np.linalg.det(s['cell'])) < 1e-08:
        raise ValueError('A full-rank 3D cell including vacuum is required.')
    info = layer_info(s, ads, float(p.get('layer_tol', 0.35)))
    layers = info['layers']
    slab = info['slab_indices']
    if not layers:
        raise ValueError('No substrate atoms selected.')
    top = parse_indices(
        p.get('surface_indices', ''),
        len(s['symbols'])
    ) or layers[0]['indices']
    if not set(top) <= set(slab):
        raise ValueError('Top-layer selection contains adsorbate atoms.')
    if max((dot(s['positions'][i], normal) for i in top)) - min((dot(s['positions'][i], normal) for i in top)) > float(p.get('layer_tol', 0.35)):
        raise ValueError('Select a flat exposed top layer. Step/oxide classification is outside this adapter.')
    pm = Structure(
        s['cell'],
        [s['symbols'][i] for i in slab],
        [s['positions'][i] for i in slab],
        coords_are_cartesian=True,
        to_unit_cell=False,
        site_properties={'surface_properties': ['surface' if i in top else 'subsurface' for i in slab]}
    )
    finder = AdsorbateSiteFinder(
        pm,
        mi_vec=normal,
        height=float(p.get('layer_tol', 0.35))
    )
    found = finder.find_adsorption_sites(
        distance=0.0,
        put_inside=True,
        symm_reduce=0.0,
        near_reduce=1e-05,
        positions=('ontop', 'bridge', 'hollow'),
        no_obtuse_hollow=True
    )
    sites = []
    nearest = []
    for q in top:
        for j in top:
            if q != j:
                d = norm(minimum_vector(sub(s['positions'][q], s['positions'][j]), s))
                if d > 0.1:
                    nearest.append(d)
    nn = min(nearest, default=min(norm(s['cell'][0]), norm(s['cell'][1])))
    for pm_kind, kind in [
        ('ontop', 'top'),
        ('bridge', 'bridge'),
        ('hollow', 'hollow_unknown')
    ]:
        for xyz in found.get(pm_kind, []):
            xyz = [float(v) for v in xyz]
            distances = []
            for i in top:
                v = minimum_vector(sub(xyz, s['positions'][i]), s)
                v = sub(v, mul(normal, dot(v, normal)))
                distances.append((norm(v), i))
            distances.sort()
            count = {'top': 1, 'bridge': 2, 'hollow_unknown': 3}[kind]
            members = [i for _, i in distances[:count]]
            label = kind
            if kind == 'hollow_unknown' and len(top) >= 3:
                ds = [v for v, _ in distances[:3]]
                triangular = max(ds) - min(ds) < max(0.06, 0.05 * nn) and abs(sum(ds) / 3 - nn / 3 ** 0.5) < 0.1 * nn
                under = []
                for layer in layers[1:3]:
                    dlist = []
                    for i in layer['indices']:
                        v = minimum_vector(sub(xyz, s['positions'][i]), s)
                        dlist.append(norm(sub(v, mul(normal, dot(v, normal)))))
                    under.append(min(dlist, default=999.0))
                if triangular and under and (under[0] < 0.18 * nn):
                    label = 'hcp'
                elif triangular and len(under) > 1 and (under[1] < 0.18 * nn):
                    label = 'fcc'
            _unique_site(sites, xyz, label, members, s)
    sites.sort(key=lambda x: (x['kind'], *[round(v, 5) for v in x['position']]))
    for j, item in enumerate(sites):
        item['site_id'] = f"{item['kind']}_{j + 1:03d}"
    return (
        sites,
        {
            'backend': 'pymatgen.AdsorbateSiteFinder',
            'site_count': len(sites),
            'normal': normal,
            'layer_count': len(layers),
            'top_layer_indices': top,
            'nearest_neighbour_A': nn,
            'layer_tolerance_A': float(p.get('layer_tol', 0.35)),
            'assumption': 'pymatgen ontop/bridge/hollow; fcc/hcp labels from flat triangular layer registry only'
        }
    )
