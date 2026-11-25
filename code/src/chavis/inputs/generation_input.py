import copy
import math
from chavis.search.geometry import add, basis, norm, rotate, sub, parse_indices, validate
from chavis.search.collision import GeometryError, dual as _dual
from chavis.search.surface_sites import _flat, _number, _unwrap_adsorbates, component
from chavis.search.connectivity import connectivity
ENGINE_VERTICAL_SECTOR_ERROR = 'Bond projection on the surface is zero. Use a 3D cone or set half-angle=180 for all azimuths.'


def vertical_sector_axis_evidence(vector, normal, *, mode, domain, half_angle):
    length = norm(vector)
    axis = [v / length for v in vector] if length > 1e-09 else [1.0, 0.0, 0.0]
    projection = norm(_flat(axis, normal))
    unsupported = domain == 'sector' and projection < 1e-07 and (mode in ('cleavage', 'association', 'transfer')) and (half_angle < 180)
    return {
        'bond_vector_A': list(vector),
        'normal': list(normal),
        'planar_projection_length': projection,
        'engine_vertical_tolerance': 1e-07,
        'unsupported': unsupported
    }


def diagnose_generation_input(parent, params):
    original_parent, original_params = (copy.deepcopy(parent), copy.deepcopy(params))
    p = copy.deepcopy(params)
    s = copy.deepcopy(validate(parent))
    _dual(s)
    n = len(s['symbols'])
    mode = p.get('mode', 'cleavage')
    ads = parse_indices(p.get('adsorbate', s.get('adsorbate_indices', [])), n)
    anchor = int(p.get('anchor', ads[0]))
    moving = parse_indices(p.get('moving', ''), n)
    if not moving:
        raise ValueError('Select at least one moving atom.')
    lead = moving[0]
    graph, backend = connectivity(
        s,
        ads,
        _number(p, 'bond_scale', 1.2, 0.9, 1.6),
        p.get('graph_backend', 'auto'),
        p.get('bond_overrides')
    )
    original_graph = {i: set(v) for i, v in graph.items()}
    unwrapped = []
    if p.get('unwrap_adsorbate', True):
        s, unwrapped = _unwrap_adsorbates(s, ads, graph)
    _, _, normal = basis(s)
    rotation = _number(p, 'rotation', 0)
    origin = s['positions'][anchor][:]
    if rotation:
        rotating = component(original_graph, anchor) if mode == 'cleavage' else set(moving)
        for i in rotating:
            s['positions'][i] = add(
                origin,
                rotate(sub(s['positions'][i], origin), normal, math.radians(rotation))
            )
    vector = sub(s['positions'][lead], s['positions'][anchor])
    half = _number(p, 'half_angle', 30, 0, 180)
    domain = p.get('domain', 'sector')
    if domain not in ('sector', 'cone3d'):
        raise GeometryError('Choose sector or cone3d domain.')
    evidence = vertical_sector_axis_evidence(
        vector,
        normal,
        mode=mode,
        domain=domain,
        half_angle=half
    )
    evidence.update({
        'kind': 'input_incompatible_vertical_sector_axis' if evidence['unsupported'] else 'supported_axis',
        'anchor': anchor,
        'moving': lead,
        'moving_fragment': moving,
        'domain': domain,
        'half_angle': half,
        'mode': mode,
        'rotation_deg': rotation,
        'periodic_unwrapped_atom_indices': unwrapped,
        'graph_backend': backend,
        'automatic_parameter_change': False,
        'engine_error': ENGINE_VERTICAL_SECTOR_ERROR if evidence['unsupported'] else None
    })
    if parent != original_parent or params != original_params:
        raise AssertionError('diagnostic mutated parent or params')
    return evidence
