from collections import deque
import copy
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from chavis.search.geometry import from_ase, to_ase, basis
from .contact_placement import contact_target
from .library_sites import enumerate_sites
from chavis.search.surface_sites import infer_graph
from chavis.search.collision import minimum_vector


def orientation_variants(positions, placeholder_binder=None, azimuth_step_deg=45.0):
    from .orientations import generate
    return [{
        'orientation': f"{o['feature']}{o['feature_index']}_az{int(o['azimuth_deg'])}",
        'feature': o['feature'],
        'feature_index': o['feature_index'],
        'feature_atoms': o['members'],
        'azimuth_deg': o['azimuth_deg'],
        'placeholder_binder': placeholder_binder,
        'azimuth_step_deg': azimuth_step_deg
    } for o in generate(
        positions,
        placeholder_binder=placeholder_binder,
        azimuth_step_deg=azimuth_step_deg
    )]


def align_vector(positions, vector, target, origin):
    v, t = (np.asarray(vector, float), np.asarray(target, float))
    v = v / np.linalg.norm(v)
    t = t / np.linalg.norm(t)
    axis = np.cross(v, t)
    c = float(np.clip(v @ t, -1, 1))
    if np.linalg.norm(axis) < 1e-12:
        if c > 0:
            return np.asarray(positions).copy()
        axis = np.cross(v, np.eye(3)[np.argmin(np.abs(v))])
        axis /= np.linalg.norm(axis)
        rotation = 2 * np.outer(axis, axis) - np.eye(3)
    else:
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + np.sqrt(1 - c * c) * skew + (1 - c) * skew @ skew
    origin = np.asarray(origin)
    return (np.asarray(positions) - origin) @ rotation.T + origin


def orient(fragment, variant, normal=(0, 0, 1)):
    from .orientations import generate, rotation_to
    a = fragment.copy()
    rows = generate(
        a.positions,
        placeholder_binder=variant.get('placeholder_binder'),
        azimuth_step_deg=variant.get('azimuth_step_deg', 45.0)
    )
    if 'feature_index' not in variant:
        variant = {
            **variant,
            'feature_index': rows[0]['feature_index'],
            'azimuth_deg': 0.0
        }
    match = [o for o in rows if o['feature_index'] == variant['feature_index'] and abs(o['azimuth_deg'] - variant['azimuth_deg']) < 1e-09]
    if len(match) != 1:
        raise ValueError('Orientation variant does not match this molecule')
    p = match[0]['positions']
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    p = p @ rotation_to([0, 0, 1], n).T
    a.positions = p
    members = match[0]['members']
    return (a, {int(i): 1.0 / len(members) for i in members})


def surface_frame(atoms):
    s = from_ase(atoms)
    ads = s['adsorbate_indices']
    slab = [i for i in range(len(atoms)) if i not in ads]
    profile = (s.get('meta') or {}).get('surface_profile')
    if isinstance(profile, dict) and profile.get('element'):
        from chavis.judgment.surface_context import selected_surface_element
        if selected_surface_element(s) != profile.get('element'):
            raise ValueError('Campaign requires a validated saved source slab profile')
    elif len(slab) != 36 or any((atoms[i].symbol != 'Ni' for i in slab)):
        raise ValueError('Campaign requires explicit Ni36 source slab')
    normal = np.asarray(basis(s)[2])
    heights = atoms.positions[slab] @ normal
    top = [i for i in slab if atoms.positions[i] @ normal > float(heights.max()) - 0.5]
    size = (profile or {}).get('size') if isinstance(profile, dict) else None
    expected = int(size[0]) * int(size[1]) if size else 9
    if len(top) != expected:
        raise ValueError(f'Cannot identify {expected} top-layer slab atoms in snapshot')
    return (normal, atoms.positions[top].mean(axis=0), top)


def site_at(atoms, kind):
    s = from_ase(atoms)
    sites, summary = enumerate_sites(
        s,
        s['adsorbate_indices'],
        {'site_backend': 'existing_terrace'}
    )
    wanted = 'top' if kind == 'ontop' else kind
    choices = [q for q in sites if q['kind'] == wanted]
    if not choices:
        raise ValueError(f'No {kind} site in preserved site backend')
    return (choices[0], summary)


def place_on_snapshot(template, fragment, weights, site):
    old = template.copy()
    ads = old.info['adsorbate_indices']
    slab_indices = [i for i in range(len(old)) if i not in ads]
    if slab_indices != list(range(36)):
        raise ValueError('Explicit slab index adapter required')
    slab = old[slab_indices]
    slab.info = {}
    a = slab + fragment
    fixed = sorted((i for con in old.constraints for i in con.get_indices()))
    if any((i >= 36 for i in fixed)):
        raise ValueError('Adsorbate constraint cannot be transferred to a new molecule')
    a.set_constraint(FixAtoms(indices=fixed))
    a.set_array('rw_id', np.arange(len(a)))
    a.info['adsorbate_indices'] = list(range(36, len(a)))
    if 'surface_profile' in old.info:
        a.info['surface_profile'] = copy.deepcopy(old.info['surface_profile'])
    q, summary = site_at(template, site)
    normal, plane, top = surface_frame(template)
    reference = sum((float(w) * a.positions[36 + i] for i, w in weights.items()))
    a.positions[36:] += np.asarray(q['position']) - reference
    lead = 36
    target_for_lead = a.positions[lead].tolist()
    target, contact = contact_target(
        from_ase(a),
        list(range(36, len(a))),
        list(range(36, len(a))),
        target_for_lead,
        normal.tolist(),
        1.0
    )
    a.positions[36:] += np.asarray(target) - a.positions[lead]
    meta = {
        'seed_site': site,
        'site_position_A': q['position'],
        'site_backend': summary,
        'reference_weights': {str(36 + i): w for i, w in weights.items()},
        'normal': normal.tolist(),
        'plane_point_A': plane.tolist(),
        'plane_top_indices': top,
        'contact': contact,
        'atom_namespace': 'campaign_new_composition',
        'slab_source_indices': slab_indices
    }
    a.info['campaign_seed'] = copy.deepcopy(meta)
    return (a, meta)


def cleavage(atoms, case, variant_index=0):
    s = from_ase(atoms)
    ads = s['adsorbate_indices']
    graph = infer_graph(s, ads, 1.2)
    first, second = case['broken_bond_elements'].split('-')
    pairs = [(i, j) for i in ads for j in sorted(graph[i]) if s['symbols'][i] == first and s['symbols'][j] == second and (first != second or i < j)]
    if not pairs:
        raise ValueError('Target bond absent in actual input graph')
    anchor, lead = pairs[variant_index % len(pairs)]
    if case['id'] == 'X02':
        anchor, lead = (lead, anchor)
    g = {k: set(v) for k, v in graph.items()}
    g[anchor].remove(lead)
    g[lead].remove(anchor)
    moving = {lead}
    pending = deque([lead])
    while pending:
        i = pending.popleft()
        for j in g[i] - moving:
            moving.add(j)
            pending.append(j)
    if anchor in moving:
        raise ValueError('Target cut does not separate a fragment')
    return (
        anchor,
        [lead] + sorted(moving - {lead}),
        sorted(((min(i, j), max(i, j)) for i in ads for j in graph[i] if i < j))
    )
