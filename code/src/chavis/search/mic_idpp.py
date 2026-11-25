import copy
import numpy as np
from ase.geometry import find_mic
from ase.mep import NEB
from ase.mep.neb import IDPP, idpp_interpolate
from ase.optimize import FIRE
from .geometry import RADII, compatible, from_ase, interpolate, to_ase, validate
from .collision import collisions as _collisions
from .policy import IDPPPolicy, LEGACY_GUI_IDPP, NI_DRM_IDPP, NORMAL_REACTION_MIC_IDPP
from chavis.runtime.numeric_defaults import load_numeric_defaults
LOCAL_GEOMETRY_POLICY = load_numeric_defaults()['search']['local_geometry']


class PathPreparationError(ValueError):

    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def _stable_fragments(start, end):
    ads = start.info.get('adsorbate_indices', [])
    edges = []
    for n, i in enumerate(ads):
        for j in ads[n + 1:]:
            cutoff = LOCAL_GEOMETRY_POLICY['surviving_bond_radius_scale'] * (RADII[start[i].symbol] + RADII[start[j].symbol])
            if start.get_distance(i, j, mic=True) < cutoff and end.get_distance(i, j, mic=True) < cutoff:
                edges.append((i, j))
    groups = [{i} for i in ads]
    for i, j in edges:
        left = next((g for g in groups if i in g))
        right = next((g for g in groups if j in g))
        if left is not right:
            left.update(right)
            groups.remove(right)
    return ([sorted(g) for g in groups], edges)


def _coherent_endpoint_images(start, end):
    groups, edges = _stable_fragments(start, end)
    aligned_start = start.copy()
    aligned_start.info = copy.deepcopy(start.info)
    aligned = end.copy()
    aligned.info = copy.deepcopy(end.info)
    adjusted = []
    for group in groups:
        if not group:
            continue
        root = group[0]
        for original, target in ((start, aligned_start), (end, aligned)):
            visited = {root}
            while len(visited) < len(group):
                progressed = False
                for i, j in edges:
                    if i in visited and j in group and (j not in visited):
                        target.positions[j] = target.positions[i] + original.get_distance(i, j, mic=True, vector=True)
                        visited.add(j)
                        progressed = True
                    elif j in visited and i in group and (i not in visited):
                        target.positions[i] = target.positions[j] + original.get_distance(j, i, mic=True, vector=True)
                        visited.add(i)
                        progressed = True
                if not progressed:
                    raise PathPreparationError(
                        'fragment_graph_unresolved',
                        {'group': group}
                    )
        displacement = np.mean(
            aligned.positions[group] - aligned_start.positions[group],
            axis=0
        )
        shortest, _ = find_mic(displacement, aligned_start.cell, aligned_start.pbc)
        shift = shortest - displacement
        aligned.positions[group] += shift
        adjusted.append({'atom_indices': group, 'lattice_shift_A': shift.tolist()})
    return (aligned_start, aligned, groups, adjusted)


def mic_collision_pairs(frame, ads, scale=LOCAL_GEOMETRY_POLICY['mic_collision_scale']):
    pairs = _collisions(frame, ads, scale)
    image = to_ase(frame)
    for offset, i in enumerate(ads):
        for j in ads[offset + 1:]:
            cutoff = scale * (RADII.get(frame['symbols'][i], 1.2) + RADII.get(frame['symbols'][j], 1.2))
            distance = float(image.get_distance(i, j, mic=True))
            if distance < cutoff:
                pairs.append([i, j, round(distance, 4), 'adsorbate_pair_mic'])
    return pairs


def prepare_legacy_idpp(
    input_structure,
    output_structure,
    count,
    *,
    policy: IDPPPolicy=LEGACY_GUI_IDPP
):
    if policy != LEGACY_GUI_IDPP:
        raise ValueError('Legacy GUI IDPP requires its named compatibility policy')
    a = validate(input_structure)
    b = validate(output_structure)
    compatible(a, b)
    raw = interpolate(a, b, count)
    images = [to_ase(s) for s in raw]
    neb = NEB(images)
    idpp_interpolate(
        neb,
        fmax=policy.fmax_eV_A,
        optimizer=FIRE,
        mic=policy.mic,
        steps=policy.steps,
        log=None,
        traj=None
    )
    frames = [from_ase(x) for x in images]
    for frame in frames:
        frame.pop('energy', None)
        frame['meta']['path_type'] = 'ASE IDPP initial guess — NOT a MEP/TS'
    return frames


def prepare_idpp(
    input_structure,
    output_structure,
    count,
    *,
    collision_scale=LOCAL_GEOMETRY_POLICY['mic_collision_scale'],
    policy: IDPPPolicy=NI_DRM_IDPP,
    path_class='local_reaction'
):
    if path_class != 'local_reaction':
        raise ValueError('LONG_DIFFUSION_PATH_NOT_SUPPORTED_EXPLICIT_LOCAL_REACTION_ONLY')
    if policy not in (NI_DRM_IDPP, NORMAL_REACTION_MIC_IDPP):
        raise ValueError('MIC IDPP requires a named normal-local reaction policy')
    'Return JSON frames and diagnostics; raise with evidence on unusable bands.'
    first = validate(input_structure)
    last = validate(output_structure)
    compatible(first, last)
    for key in ('fixed_indices', 'adsorbate_indices', 'initial_magmoms'):
        if first.get(key) != last.get(key):
            raise ValueError(f'{key} differs between IDPP endpoints')
    if first.get('meta', {}).get('charge') != last.get('meta', {}).get('charge'):
        raise ValueError('Charge context differs between IDPP endpoints')
    if not LOCAL_GEOMETRY_POLICY['image_count_min'] <= count <= LOCAL_GEOMETRY_POLICY['image_count_max']:
        raise ValueError('Use 3 to 40 total images, including endpoints.')
    fixed = first.get('fixed_indices', [])
    if fixed and (not np.allclose(
        np.asarray(first['positions'])[fixed],
        np.asarray(last['positions'])[fixed],
        atol=1e-06,
        rtol=0
    )):
        raise ValueError('Fixed atom coordinates differ between IDPP endpoints')
    start, end = (to_ase(first), to_ase(last))
    aligned_start, aligned, groups, shifts = _coherent_endpoint_images(start, end)
    images = [aligned_start.copy() for _ in range(count)]
    images[-1] = aligned
    neb = NEB(images, method='improvedtangent')
    neb.interpolate(method='linear', mic=True, apply_constraint=False)
    linear = [from_ase(a) for a in images]
    for a in images:
        a.calc = None
    displacement = np.asarray([a.positions - images[0].positions for a in images])
    fragment_checks = []
    for group in groups:
        if len(group) < 2:
            continue
        reference = displacement[-1, group] - displacement[0, group]
        for k in range(1, count - 1):
            ideal = k / (count - 1) * reference
            error = np.linalg.norm(
                displacement[k, group] - ideal - (displacement[k, group[0]] - ideal[0]),
                axis=1
            )
            fragment_checks.append({
                'image': k,
                'atom_indices': group,
                'max_relative_interpolation_error_A': float(max(error))
            })
    diagnostics = {
        'method': 'ASE_LINEAR_MIC_IDPP_MIC',
        'image_count': count,
        'fragment_image_shifts': shifts,
        'fragment_checks': fragment_checks,
        'idpp': {
            'optimizer': 'FIRE',
            'steps_limit': policy.steps,
            'surrogate_force_target_eV_A': policy.fmax_eV_A,
            'objective': 'ASE IDPP pair-distance surrogate; not a physical energy'
        }
    }
    if any((v['max_relative_interpolation_error_A'] > LOCAL_GEOMETRY_POLICY['fragment_image_consistency_atol_A'] for v in fragment_checks)):
        raise PathPreparationError('fragment_periodic_image_conflict', diagnostics)
    d0 = images[0].get_all_distances(mic=True)
    d1 = images[-1].get_all_distances(mic=True)
    original_calcs = [a.calc for a in images]
    opt = None
    try:
        for k, a in enumerate(images):
            a.calc = IDPP(d0 + k * (d1 - d0) / (count - 1), mic=policy.mic)
        with FIRE(neb, logfile=None, trajectory=None) as opt:
            converged = bool(opt.run(fmax=policy.fmax_eV_A, steps=policy.steps))
        residual = float(np.linalg.norm(neb.get_forces(), axis=1).max())
        objectives = [float(a.get_potential_energy()) for a in images]
        diagnostics['idpp'].update(
            converged=converged,
            steps=opt.nsteps,
            surrogate_force_residual_eV_A=residual,
            objective_total=sum(objectives[1:-1]),
            objective_per_image=objectives
        )
    finally:
        for a, calc in zip(images, original_calcs):
            a.calc = calc
    images[0].positions[:] = start.positions
    images[-1].positions[:] = end.positions
    if fixed:
        for image in images[1:-1]:
            image.positions[fixed] = start.positions[fixed]
    frames = [from_ase(a) for a in images]
    ads = first.get('adsorbate_indices', [])
    collisions = []
    for k, frame in enumerate(frames[1:-1], 1):
        pairs = mic_collision_pairs(frame, ads, collision_scale)
        if pairs:
            collisions.append({'image': k, 'pairs': pairs})
    diagnostics['mic_collision_scale'] = collision_scale
    diagnostics['interior_collisions'] = collisions
    diagnostics['usable'] = not collisions
    if collisions:
        raise PathPreparationError('idpp_interior_mic_collision', diagnostics)
    for k, frame in enumerate(frames):
        frame.pop('energy', None)
        frame['meta']['path_type'] = 'ASE linear MIC + IDPP MIC initial guess — NOT a MEP/TS'
        frame['meta']['image'] = k
    diagnostics['linear_initial_frames'] = linear
    return (frames, diagnostics)
