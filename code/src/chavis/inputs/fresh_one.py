from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
from chavis.runtime.hashing import sha256_file
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from chavis.runtime.numeric_defaults import load_numeric_defaults
from chavis.runtime.telemetry import StageClock
from chavis.search.geometry import from_ase, to_ase
from chavis.search.collision import GeometryError
from chavis.search.ts_helpers import ads_edges, assert_slab
from chavis.search.surface_sites import generate_surface
from chavis.judgment.policy import FS_RELAXATION_TRIAL_CAP_DEFAULT, reaction_role_contract
from .fragment import build_fragment, finalize_direction
from .fs_sites import direct_fs_seeds
from .seed_geometry import orient, orientation_variants, place_on_snapshot, surface_frame, cleavage
from .new_endpoints import _relax
AVAILABILITY = 'prepared_fresh_IS_FS_pair'


class PreparationStopped(RuntimeError):
    pass


class PreparationIncomplete(RuntimeError):
    REASONS = frozenset({
        'gas_uma_not_converged',
        'no_valid_FS_in_declared_existing_trials',
        'FS_target_pair_zero_distance',
        'GAS_BOUNDARY_UNRESOLVED',
        'FS_not_separated_for_reverse_scan',
        'reverse_scan_zero_pair_vector',
        'reverse_scan_point_not_converged',
        'reverse_scan_IS_not_converged',
        'reverse_scan_did_not_recombine_requested_molecule',
        'fresh_bound_parent_molecular_IS_not_valid',
        'fresh_requested_parent_molecular_IS_not_valid',
        'FS_aligned_IS_not_stable_requested_molecule',
        'scan_first_IS_not_valid_requested_molecule',
        'fresh_endpoint_target_graph_mismatch',
        'fresh_FS_target_bond_not_separated',
        'surface_sector_undefined_for_vertical_bond'
    })

    def __init__(self, reason_code):
        if reason_code not in self.REASONS:
            raise ValueError(f'Unregistered preparation-unconfirmed reason: {reason_code}')
        self.reason_code = reason_code
        super().__init__(reason_code)


class PreparationQualityError(RuntimeError):
    """"""


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return sha256_file(path)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8'
    )
    temporary.replace(path)


def _edge_list(atoms):
    return sorted([sorted(pair) for pair in ads_edges(atoms)])


def _clock(stage_id, rid, factory, *, origin, inputs=()):
    return StageClock.start(
        stage_id,
        rid,
        factory,
        prep_origin=origin,
        input_artifact_ids=inputs
    )


def _finish_stage(
    output,
    clock,
    factory,
    *,
    status='completed',
    reason=None,
    artifacts=()
):
    event = clock.finish(
        factory,
        status=status,
        stop_reason=reason,
        output_artifact_ids=artifacts
    )
    _save(Path(output) / 'stage_telemetry.json', event)
    return event


def _gas_relax(fragment, factory, output, check):
    from ase import Atoms
    from ase.optimize import BFGS
    from fairchem.core import FAIRChemCalculator
    defaults = load_numeric_defaults()['inputs']
    work = Atoms(fragment.raw_symbols, positions=fragment.raw_positions)
    work.pbc = False
    work.info.update(charge=fragment.charge, spin=fragment.spin)
    work.calc = FAIRChemCalculator(factory.predictor, task_name='omol')
    before = work.positions.copy()
    e_before = float(work.get_potential_energy())
    opt = BFGS(work, logfile=str(Path(output) / 'gas.log'))
    opt.attach(check, interval=1)
    ok = bool(opt.run(
        fmax=float(defaults['gas_fmax']),
        steps=int(defaults['gas_steps'])
    ))
    e_after = float(work.get_potential_energy())
    result = {
        'converged': ok,
        'steps': int(opt.nsteps),
        'e_before': e_before,
        'e_after': e_after,
        'moved_max_A': float(np.linalg.norm(work.positions - before, axis=1).max()),
        'symbols': work.get_chemical_symbols(),
        'positions': work.positions.tolist(),
        'charge': fragment.charge,
        'spin': fragment.spin,
        'model_task': 'omol',
        'log_path': str(Path(output) / 'gas.log')
    }
    _save(Path(output) / 'gas_optimized.json', result)
    fragment.raw_positions = work.positions.tolist()
    fragment.positions = work.positions.tolist()
    work.calc = None
    if not ok:
        raise PreparationIncomplete('gas_uma_not_converged')
    return result


def _validate_input(entry, species, reaction, slab):
    rid = entry['reaction_id']
    schema = entry.get('schema')
    cu = schema in (
        'cu_drm_fresh_preparation_entry_v1',
        'cu_drm_fresh_preparation_entry_v3'
    )
    ni100 = schema in (
        'ni100_fresh_preparation_entry_v1',
        'ni100_fresh_preparation_entry_v3'
    )
    if schema not in (
        'cu_drm_fresh_preparation_entry_v1',
        'cu_drm_fresh_preparation_entry_v3',
        'ni100_fresh_preparation_entry_v1',
        'ni100_fresh_preparation_entry_v3',
        'ni_drm_fresh_preparation_entry_v1',
        'ni_drm_fresh_preparation_entry_v3'
    ):
        raise ValueError('Frozen recipe schema unsupported')
    v3 = schema.endswith('_v3')
    expected_schema = ('cu_drm' if cu else 'ni100' if ni100 else 'ni_drm') + ('_fresh_preparation_entry_v3' if v3 else '_fresh_preparation_entry_v1')
    if entry.get('schema') != expected_schema or reaction.get('id') != rid or reaction.get('parent_species') != entry.get('species') or (not reaction.get('enabled')) or (entry.get('model') != 'uma-s-1p2p1') or (entry.get('task') != 'oc20'):
        raise ValueError('Frozen recipe identity/model/task mismatch')
    if any((k in entry for k in (
        'is_structure',
        'fs_structure',
        'endpoint_record',
        'parent_guess'
    ))):
        raise ValueError('Sanitized recipe contains an old adsorbate endpoint')
    metal = 'Cu' if cu else 'Ni'
    slab_schema = 'cu_drm_gui_cu36_substrate_v1' if cu else 'ni100_ase_ni36_substrate_v1' if ni100 else 'ni_drm_external_ni36_substrate_v1'
    if slab.get('schema') != slab_schema or slab.get('for_reaction') != rid or slab.get('symbols') != [metal] * 36 or (slab.get('atom_ids') != list(range(36))) or (slab.get('fixed_indices') != list(range(18))) or (slab.get('optimized_adsorbate_coordinates_included') is not False):
        raise ValueError('Declared 36-atom substrate identity mismatch')
    if cu:
        from chavis.judgment.surface_context import selected_surface_element
        if selected_surface_element(slab) != 'Cu':
            raise ValueError('Validated GUI Cu(111) substrate profile required')
        if slab.get('gui_surface_job_id') is None or slab.get('surface_profile') != (slab.get('meta') or {}).get('surface_profile'):
            raise ValueError('GUI Cu substrate provenance/profile mismatch')
    if ni100:
        from chavis.judgment.surface_context import selected_surface_element
        profile = slab.get('surface_profile')
        if selected_surface_element(slab) != 'Ni' or not isinstance(profile, dict) or profile.get('facet') != 'fcc100' or (profile != (slab.get('meta') or {}).get('surface_profile')):
            raise ValueError('Validated fresh Ni(100) substrate profile required')
    if not isinstance(slab.get('surface_profile'), dict) or slab.get('surface_profile') != (slab.get('meta') or {}).get('surface_profile'):
        raise ValueError('Fresh slab surface_profile missing or inconsistent')
    if v3:
        expected_criterion = 'cu111-reaction-success-v3' if cu else 'ni100-reaction-success-v3' if ni100 else 'ni-drm-reaction-success-v3'
        expected_facet = 'fcc100' if ni100 else 'fcc111'
        if entry.get('criterion_profile') != 'v3' or entry.get('criterion_version') != expected_criterion or entry.get('surface_profile') != slab['surface_profile'] or (entry.get('surface_facet') != expected_facet) or (slab['surface_profile'].get('facet') != expected_facet):
            raise ValueError('Frozen v3 criterion/surface contract mismatch')
    if species.get('smiles') is None or species.get('mode') not in ('Cl', 'free:C', 'free:O', 'free:H'):
        raise ValueError('Named molecular recipe unsupported')
    if entry.get('seed', {}).get('site') not in (('ontop', 'bridge', 'fourfold') if ni100 else ('fcc', 'hcp', 'bridge', 'ontop')):
        raise ValueError('Unspecified existing site type')
    reaction_role_contract(
        rid,
        declared=entry.get('allowed_parent_roles'),
        equation=reaction.get('equation'),
        requested_direction=entry.get('requested_direction', 'dissociation')
    )
    if not isinstance(entry.get('settings'), dict):
        raise ValueError('Missing original endpoint settings')
    if entry.get('endpoint_preparation_strategy') not in (
        'PARENT_BASELINE',
        'REVERSE_BOND_SCAN',
        'NEW_APPROVED_PARENT_FIRST_DIRECT_COADSORBED_FS'
    ):
        raise ValueError('Unknown endpoint preparation strategy')
    return metal


def _coadsorbed_rows(molecular, entry, reaction, moving, anchor):
    settings = entry['settings']
    strategy = entry['endpoint_preparation_strategy']
    if strategy in (
        'REVERSE_BOND_SCAN',
        'NEW_APPROVED_PARENT_FIRST_DIRECT_COADSORBED_FS'
    ):
        rows, metadata = direct_fs_seeds(
            molecular,
            moving,
            anchor,
            settings,
            entry['seed']['site']
        )
        for row in rows:
            row['fragment_site_kinds'] = {
                'retained_parent_seed': entry['seed']['site'],
                'moving_candidate': row['site']['kind']
            }
        return (rows, {'method': 'existing_direct_coadsorbed_FS', **metadata})
    params = dict(
        anchor=anchor,
        moving=moving,
        adsorbate=molecular.info['adsorbate_indices'],
        height_mode='drop',
        contact_scale=1.0,
        collision_scale=0.7,
        domain='sector',
        half_angle=settings['half_angle'],
        rmin=settings['rmin'],
        rmax=settings['rmax']
    )
    full_azimuth_fallback = False
    try:
        rows = generate_surface(from_ase(molecular), params)
    except GeometryError as exc:
        if str(exc) == 'Bond projection on the surface is zero. Use a 3D cone or set half-angle=180 for all azimuths.':
            params['half_angle'] = 180.0
            try:
                rows = generate_surface(from_ase(molecular), params)
            except GeometryError as fallback_exc:
                raise PreparationIncomplete('surface_sector_undefined_for_vertical_bond') from fallback_exc
            full_azimuth_fallback = True
        else:
            raise
    accepted = [row for row in rows if row['accepted']]
    for row in accepted:
        row['fragment_site_kinds'] = {
            'retained_parent_seed': entry['seed']['site'],
            'moving_candidate': row['detail']['site']
        }
    offset = int(entry['seed'].get('candidate_offset', 0))
    if accepted:
        offset %= len(accepted)
        accepted = accepted[offset:] + accepted[:offset]
    return (
        accepted,
        {
            'method': 'existing_surface_sector_FS',
            'parameters': params,
            'full_azimuth_vertical_bond_fallback': full_azimuth_fallback,
            'declared_half_angle_deg': settings['half_angle'],
            'enumerated': len(rows),
            'accepted': len(accepted)
        }
    )


def _parent_variant_grid(entry, species, slab, fragment):
    from chavis.search.surface_sites import enumerate_terrace_sites
    profile = slab.get('surface_profile') or (slab.get('meta') or {}).get('surface_profile')
    if not isinstance(profile, dict) or not profile.get('facet'):
        raise ValueError('FRESH_SURFACE_PROFILE_MISSING_OR_UNSUPPORTED')
    structure = deepcopy(slab)
    structure['adsorbate_indices'] = []
    structure.setdefault('meta', {})['surface_profile'] = profile
    found, _ = enumerate_terrace_sites(structure, [])
    sites = sorted({'ontop' if s['kind'] == 'top' else s['kind'] for s in found})
    binder = fragment.binder if species.get('mode') == 'Cl' else None
    return [{
        **entry['seed'],
        **orientation,
        'site': site
    } for site in sites for orientation in orientation_variants(
        fragment.positions,
        placeholder_binder=binder
    )]


def _build_parent_seed(
    fragment,
    slab,
    entry,
    species,
    output,
    *,
    variant=None,
    finalized=False,
    artifact_name='molecular_seed.json'
):
    from ase import Atoms
    if not finalized:
        finalize_direction(fragment)
    variant = variant or entry['seed']
    molecular = Atoms(fragment.symbols, positions=fragment.positions)
    if 'Cl' in molecular.get_chemical_symbols():
        raise ValueError('Removable Cl remained in physical structure')
    substrate = deepcopy(slab)
    substrate['adsorbate_indices'] = []
    template = to_ase(substrate)
    assert len(template) == 36
    normal, _, _ = surface_frame(template)
    oriented, weights = orient(molecular, variant, normal)
    parent, placement = place_on_snapshot(template, oriented, weights, variant['site'])
    parent.info.update(charge=fragment.charge, spin=fragment.spin)
    if parent.get_chemical_symbols() != entry['expected_symbols'] or list(parent.arrays['rw_id']) != list(range(len(parent))):
        raise ValueError('Fresh molecular atom composition/order changed')
    assert_slab(parent)
    _save(
        Path(output) / artifact_name,
        {
            'structure': from_ase(parent),
            'placement': placement,
            'orientation': variant,
            'source': 'fresh_smiles_on_GUI_Cu111_clean_slab' if slab['symbols'][0] == 'Cu' else 'fresh_smiles_on_new_Ni100_ase_slab' if slab.get('surface_profile', {}).get('facet') == 'fcc100' else 'fresh_smiles_on_external_Ni_only_substrate'
        }
    )
    return parent


def _apply_selected_site(entry, variant):
    entry['seed'] = variant
    entry['settings']['site'] = variant['site']
    entry.setdefault('config', {}).setdefault('settings', {})['site'] = variant['site']


def _select_parent_seed(
    candidates,
    entry,
    factory,
    output,
    check,
    *,
    allow_none=False,
    criterion_profile='v2'
):
    from chavis.judgment.graph import _endpoint
    attempts = []
    chosen = None
    fallback = None
    contract = reaction_role_contract(
        entry['reaction_id'],
        declared=entry.get('allowed_parent_roles')
    ) if criterion_profile == 'v3' else None
    allowed_roles = set(contract['allowed_parent_roles']) if contract is not None else {'surface_contact', 'weak_contact'}
    for index, (seed, variant) in enumerate(candidates):
        check()
        before_calls = getattr(factory, 'calls', None)
        started = time.monotonic()
        started_cpu = time.process_time()
        try:
            final, diagnostic = _relax(
                seed,
                seed,
                entry['settings'],
                output,
                f'molecular_IS_{index + 1:02}',
                factory,
                check
            )
        except Exception as exc:
            after_calls = getattr(factory, 'calls', None)
            attempts.append({
                'candidate_index': index,
                'site': variant['site'],
                'orientation': variant['orientation'],
                'variant': variant,
                'energy_eV': None,
                'converged': None,
                'finite': None,
                'parent_role': None,
                'valid': False,
                'rejection_reason': f'{type(exc).__name__}: {exc}',
                'status': 'operational_error',
                'wall_monotonic_s': time.monotonic() - started,
                'process_cpu_s': time.process_time() - started_cpu,
                'model_calls': after_calls - before_calls if before_calls is not None and after_calls is not None else None
            })
            _save(
                Path(output) / 'molecular_IS_candidates.json',
                {
                    'candidates': attempts,
                    'selected_index': chosen[0] if chosen else None,
                    'allowed_parent_roles': sorted(allowed_roles)
                }
            )
            raise
        edges = diagnostic['adsorbate_edges']
        graph_valid = edges == entry['expected_IS_edges']
        role = None
        if diagnostic['finite']:
            refs = {tuple(pair): float(seed.get_distance(*pair, mic=True)) for pair in entry['expected_IS_edges']}
            components = _endpoint(
                from_ase(final),
                refs,
                criterion_profile=criterion_profile
            )['components']
            if len(components) == 1:
                role = components[0]['role']
        role_valid = role in allowed_roles
        fallback_valid = contract is not None and contract['detached_fallback_separate_channel'] and (role == 'detached_molecule')
        valid = bool(diagnostic['converged'] and diagnostic['finite'] and graph_valid and role_valid and math.isfinite(diagnostic['energy_eV']))
        reason = None if valid else 'not_converged' if not diagnostic['converged'] else 'nonfinite_result' if not diagnostic['finite'] else 'molecular_graph_changed' if not graph_valid else 'detached_parent_requires_separate_channel' if fallback_valid else 'requested_parent_role_mismatch' if contract is not None else 'intended_surface_attachment_role_lost'
        after_calls = getattr(factory, 'calls', None)
        attempt = {
            'candidate_index': index,
            'site': variant['site'],
            'orientation': variant['orientation'],
            'variant': variant,
            'energy_eV': diagnostic['energy_eV'],
            'converged': diagnostic['converged'],
            'finite': diagnostic['finite'],
            'actual_edges': edges,
            'expected_edges': entry['expected_IS_edges'],
            'parent_role': role,
            'valid': valid,
            'fallback_channel_eligible': bool(fallback_valid and diagnostic['converged'] and diagnostic['finite'] and graph_valid),
            'rejection_reason': reason,
            'wall_monotonic_s': time.monotonic() - started,
            'process_cpu_s': time.process_time() - started_cpu,
            'model_calls': after_calls - before_calls if before_calls is not None and after_calls is not None else None,
            'result_path': str(Path(output) / f'molecular_IS_{index + 1:02}.json')
        }
        attempts.append(attempt)
        if valid and (chosen is None or diagnostic['energy_eV'] < chosen[2]['energy_eV']):
            chosen = (index, final, diagnostic, seed, variant)
        if attempt['fallback_channel_eligible'] and math.isfinite(diagnostic['energy_eV']) and (fallback is None or diagnostic['energy_eV'] < fallback[2]['energy_eV']):
            fallback = (index, final, diagnostic, seed, variant)
        _save(
            Path(output) / 'molecular_IS_candidates.json',
            {
                'candidates': attempts,
                'selected_index': chosen[0] if chosen else fallback[0] if fallback else None,
                'selected_role': attempts[chosen[0] if chosen else fallback[0]]['parent_role'] if chosen or fallback else None,
                'selected_channel': 'requested_primary' if chosen else 'detached_parent_fallback' if fallback else None,
                'allowed_parent_roles': contract['allowed_parent_roles'] if contract is not None else ['surface_contact', 'weak_contact'],
                'selection_rule': 'lowest_energy_converged_v3_graph_and_requested_role' if contract is not None else 'lowest_energy_converged_v2_graph_and_surface_role'
            }
        )
    if chosen is None:
        if fallback is not None:
            index, final, diagnostic, seed, variant = fallback
            return (
                index,
                final,
                {
                    **diagnostic,
                    'selected_parent_role': attempts[index]['parent_role'],
                    'selected_parent_channel': 'detached_parent_fallback'
                },
                seed,
                variant
            )
        if allow_none:
            return None
        raise PreparationIncomplete('fresh_requested_parent_molecular_IS_not_valid' if contract is not None else 'fresh_bound_parent_molecular_IS_not_valid')
    index, final, diagnostic, seed, variant = chosen
    return (
        index,
        final,
        {
            **diagnostic,
            'selected_parent_role': attempts[index]['parent_role'],
            'selected_parent_channel': 'requested_primary'
        },
        seed,
        variant
    )


def _candidate_fs(parent, entry, reaction, factory, output, check):
    settings = entry['settings']
    anchor, moving, expected = cleavage(
        parent,
        reaction,
        int(entry['seed'].get('bond_variant', 0))
    )
    if [anchor, moving[0]] != [entry['anchor'], entry['moving']] or sorted([list(v) for v in expected]) != entry['expected_IS_edges']:
        raise ValueError('Fresh target atom/graph mapping differs from declared reaction')
    rows, metadata = _coadsorbed_rows(parent, entry, reaction, moving, anchor)
    enumerated = int(metadata.get('enumerated', len(rows)))
    eligible = sum((bool(row['accepted']) for row in rows))
    _save(
        Path(output) / 'FS_candidates.json',
        {
            'metadata': metadata,
            'candidate_count': len(rows),
            'enumerated_count': enumerated,
            'accepted_count': eligible,
            'geometry_filter_rejected_count': enumerated - eligible,
            'candidates': rows
        }
    )
    attempts = []
    maximum = int(settings.get(
        'max_fs_relaxation_trials',
        FS_RELAXATION_TRIAL_CAP_DEFAULT
    ))
    if maximum < 1 or maximum > FS_RELAXATION_TRIAL_CAP_DEFAULT:
        raise ValueError('FS_RELAXATION_TRIAL_CAP_OUT_OF_RANGE')
    chosen = None
    relaxed = 0

    def save_progress():
        _save(
            Path(output) / 'FS_progress.json',
            {
                'attempts': attempts,
                'selected_index': chosen[2] if chosen else None,
                'enumerated_count': enumerated,
                'accepted_count': eligible,
                'geometry_filter_rejected_count': enumerated - eligible,
                'relaxed_count': relaxed,
                'relaxation_cap': maximum,
                'exhaustion_reason': None if chosen else f'exhausted_after_{relaxed}' if relaxed == min(maximum, eligible) else None
            }
        )
    save_progress()
    for index, row in enumerate(rows):
        check()
        state = {
            'candidate_index': index,
            'accepted': bool(row['accepted']),
            'status': 'not_attempted',
            'fragment_site_kinds': row.get('fragment_site_kinds')
        }
        if not row['accepted']:
            state['status'] = 'rejected_existing_geometry_predicate'
        elif relaxed >= maximum:
            state['status'] = 'declared_trial_limit'
        elif chosen is not None:
            state['status'] = 'first_valid_FS_already_selected'
        else:
            before_calls = getattr(factory, 'calls', None)
            started = time.monotonic()
            started_cpu = time.process_time()
            try:
                trial = to_ase(row['structure'])
                final, diagnostic = _relax(
                    trial,
                    parent,
                    settings,
                    output,
                    f'FS_trial_{index + 1:02}',
                    factory,
                    check
                )
            except Exception as exc:
                after_calls = getattr(factory, 'calls', None)
                state.update(
                    status='operational_error',
                    rejection_reason=f'{type(exc).__name__}: {exc}',
                    wall_monotonic_s=time.monotonic() - started,
                    process_cpu_s=time.process_time() - started_cpu,
                    model_calls=after_calls - before_calls if before_calls is not None and after_calls is not None else None
                )
                attempts.append(state)
                save_progress()
                raise
            relaxed += 1
            valid = diagnostic['converged'] and diagnostic['finite'] and (diagnostic['adsorbate_edges'] == entry['expected_FS_edges'])
            reason = None if valid else 'not_converged' if not diagnostic['converged'] else 'nonfinite_result' if not diagnostic['finite'] else 'target_product_graph_not_retained'
            after_calls = getattr(factory, 'calls', None)
            state.update(
                status='accepted_FS' if valid else 'relaxed_invalid_FS',
                rejection_reason=reason,
                converged=diagnostic['converged'],
                fmax=diagnostic['fmax'],
                energy_eV=diagnostic['energy_eV'],
                actual_edges=diagnostic['adsorbate_edges'],
                wall_monotonic_s=time.monotonic() - started,
                process_cpu_s=time.process_time() - started_cpu,
                model_calls=after_calls - before_calls if before_calls is not None and after_calls is not None else None,
                result_path=str(Path(output) / f'FS_trial_{index + 1:02}.json')
            )
            if valid:
                chosen = (final, diagnostic, index)
        attempts.append(state)
        save_progress()
    if chosen is None:
        raise PreparationIncomplete('no_valid_FS_in_declared_existing_trials')
    return (anchor, moving, chosen)


def _aligned_is(final, parent_seed, anchor, moving, entry, factory, output, check):
    initial = final.copy()
    vector = initial.get_distance(anchor, moving[0], mic=True, vector=True)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise PreparationIncomplete('FS_target_pair_zero_distance')
    distance = parent_seed.get_distance(anchor, moving[0], mic=True)
    initial.positions[moving] += initial.positions[anchor] + distance * vector / length - initial.positions[moving[0]]
    _save(
        Path(output) / 'FS_aligned_molecular_seed.json',
        {
            'structure': from_ase(initial),
            'reference_bond_distance_A': float(distance),
            'not_an_optimized_endpoint': True
        }
    )
    is_atoms, diagnostic = _relax(
        initial,
        parent_seed,
        entry['settings'],
        output,
        'molecular_IS',
        factory,
        check
    )
    valid = diagnostic['converged'] and diagnostic['finite'] and (diagnostic['adsorbate_edges'] == entry['expected_IS_edges'])
    _save(
        Path(output) / 'molecular_IS_validation.json',
        {**diagnostic, 'valid_requested_molecular_graph': valid}
    )
    return (is_atoms, diagnostic, valid)


def _constrained_scan_point(
    seed,
    anchor,
    lead,
    distance,
    parent_seed,
    entry,
    factory,
    output,
    index,
    check
):
    from ase.constraints import FixAtoms, FixBondLengths
    from ase.optimize import BFGS
    from chavis.search.structures import ModelViewCalculator
    work = seed.copy()
    fixed = assert_slab(work)
    fixed_positions = work.positions[fixed].copy()
    work.set_constraint([
        FixAtoms(indices=fixed),
        FixBondLengths([(anchor, lead)], bondlengths=[float(distance)])
    ])
    work.calc = ModelViewCalculator(factory(), parent_seed, check=check)
    logfile = Path(output) / f'reverse_scan_{index:02}.log'
    opt = BFGS(work, logfile=str(logfile), maxstep=0.15)
    opt.attach(check, interval=1)
    converged = bool(opt.run(
        fmax=float(entry['settings']['endpoint_fmax']),
        steps=int(entry['settings']['endpoint_steps'])
    ))
    energy = float(work.get_potential_energy())
    projected_fmax = float(np.linalg.norm(work.get_forces(), axis=1).max())
    if not np.array_equal(fixed_positions, work.positions[fixed]):
        raise ValueError('Fixed substrate atoms moved during reverse bond scan')
    work.set_constraint(FixAtoms(indices=fixed))
    work.calc = None
    diagnostic = {
        'distance_A': float(distance),
        'pair_indices': [anchor, lead],
        'converged': converged,
        'steps': int(opt.nsteps),
        'energy_eV': energy,
        'projected_fmax': projected_fmax,
        'scan_constraint': 'ASE FixBondLengths; released before physical endpoint',
        'structure': from_ase(work)
    }
    _save(Path(output) / f'reverse_scan_{index:02}.json', diagnostic)
    return (work, diagnostic)


def _gas_boundary(molecular, parent_seed, factory, output, check):
    from chavis.search.structures import ModelViewCalculator
    normal, plane, _ = surface_frame(molecular)
    ads = molecular.info['adsorbate_indices']
    weights = {i: 1.0 / len(ads) for i in ads}
    reference = sum((w * molecular.positions[i] for i, w in weights.items()))

    def sample(work):
        check()
        work.calc = ModelViewCalculator(factory(), parent_seed, check=check)
        energy = float(work.get_potential_energy())
        force = float(np.sum(work.get_forces()[ads] @ normal))
        work.calc = None
        return (energy, force)
    probes = []
    for shift in ([0, 0, 0], [0, 0, 0], [0.123, 0.071, 0.037], [-0.091, 0.113, -0.043]):
        test = molecular.copy()
        test.positions += np.asarray(shift)
        energy, force = sample(test)
        probes.append({
            'whole_system_translation_A': shift,
            'energy_eV': energy,
            'molecular_normal_force_eV_A': force
        })
    espread = max((v['energy_eV'] for v in probes)) - min((v['energy_eV'] for v in probes))
    fspread = max((v['molecular_normal_force_eV_A'] for v in probes)) - min((v['molecular_normal_force_eV_A'] for v in probes))
    if max(espread, fspread) > 0.001:
        raise PreparationQualityError('gas_boundary_numerical_calibration_unstable')
    etol = max(espread * 10, np.spacing(abs(probes[0]['energy_eV'])) * 100, 1e-08)
    ftol = max(
        fspread * 10,
        np.spacing(max(abs(probes[0]['molecular_normal_force_eV_A']), 1)) * 100,
        1e-08
    )
    calibration = {
        'probes': probes,
        'energy_repeatability_eV': espread,
        'force_repeatability_eV_A': fspread,
        'energy_tolerance_eV': etol,
        'force_tolerance_eV_A': ftol,
        'tolerance_rule': '10x observed spread, 100 ULP or 1e-8 floor'
    }
    _save(Path(output) / 'gas_boundary_calibration.json', calibration)
    slab = [i for i in range(len(molecular)) if i not in ads]
    cellheight = float(molecular.cell[2] @ normal)
    top = float(np.max(molecular.positions[slab] @ normal))
    bottom = float(np.min(molecular.positions[slab] @ normal))
    rows = []
    for lift in np.arange(2.0, 13.0, 1.0):
        check()
        work = molecular.copy()
        work.positions[ads] += normal * lift
        maxheight = float(np.max(work.positions[ads] @ normal))
        image_clearance = bottom + cellheight - maxheight
        if image_clearance < 4.0 or maxheight > cellheight - 1.0:
            _save(
                Path(output) / 'gas_boundary_scan_stop.json',
                {
                    'lift_A': float(lift),
                    'opposite_slab_clearance_A': image_clearance,
                    'maximum_molecule_height_A': maxheight,
                    'reason': 'unchanged_cell_clearance_guard'
                }
            )
            break
        energy, force = sample(work)
        row = {
            'lift_A': float(lift),
            'height_A': float((reference + normal * lift - plane) @ normal),
            'energy_eV': energy,
            'molecular_normal_force_eV_A': force,
            'opposite_slab_clearance_A': image_clearance
        }
        rows.append(row)
        _save(Path(output) / 'gas_boundary_samples.json', rows)
        if len(rows) >= 3:
            recent = rows[-3:]
            if max((v['energy_eV'] for v in recent)) - min((v['energy_eV'] for v in recent)) <= etol and max((abs(v['molecular_normal_force_eV_A']) for v in recent)) <= ftol:
                work.calc = ModelViewCalculator(factory(), parent_seed, check=check)
                verified_energy = float(work.get_potential_energy())
                result = {
                    'converged': None,
                    'gas_boundary_verified': True,
                    'energy_eV': verified_energy,
                    'evidence': recent,
                    'calibration': calibration,
                    'structure': from_ase(work)
                }
                _save(Path(output) / 'gas_boundary_IS.json', result)
                return (work, result)
    raise PreparationIncomplete('GAS_BOUNDARY_UNRESOLVED')


def _reverse_scan_is(final, parent_seed, anchor, moving, entry, factory, output, check):
    lead = moving[0]
    distance0 = float(final.get_distance(anchor, lead, mic=True))
    bonded = float(parent_seed.get_distance(anchor, lead, mic=True))
    if distance0 <= bonded:
        raise PreparationIncomplete('FS_not_separated_for_reverse_scan')
    current = final.copy()
    rows = []
    for index, distance in enumerate(np.linspace(
        distance0,
        bonded,
        int(entry['settings']['images'])
    )):
        vector = current.get_distance(anchor, lead, mic=True, vector=True)
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            raise PreparationIncomplete('reverse_scan_zero_pair_vector')
        current.positions[moving] += current.positions[anchor] + float(distance) * vector / length - current.positions[lead]
        current, row = _constrained_scan_point(
            current,
            anchor,
            lead,
            distance,
            parent_seed,
            entry,
            factory,
            output,
            index,
            check
        )
        rows.append({k: v for k, v in row.items() if k != 'structure'})
        _save(Path(output) / 'reverse_scan_profile.json', rows)
        if not row['converged']:
            raise PreparationIncomplete('reverse_scan_point_not_converged')
    is_atoms, diagnostic = _relax(
        current,
        parent_seed,
        entry['settings'],
        output,
        'reverse_scan_IS',
        factory,
        check
    )
    if not diagnostic['converged']:
        raise PreparationIncomplete('reverse_scan_IS_not_converged')
    if diagnostic['adsorbate_edges'] == entry['expected_IS_edges']:
        return (is_atoms, diagnostic)
    if _edge_list(current) != entry['expected_IS_edges']:
        raise PreparationIncomplete('reverse_scan_did_not_recombine_requested_molecule')
    return _gas_boundary(current, parent_seed, factory, output, check)


def prepare_one(
    catalog_entry_path,
    species_recipe_path,
    reaction_recipe_path,
    slab_reference_path,
    output,
    *,
    factory_builder=None,
    gas_relaxer=None,
    endpoint_relaxer=None
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        'catalog_entry': Path(catalog_entry_path),
        'species_recipe': Path(species_recipe_path),
        'reaction_recipe': Path(reaction_recipe_path),
        'slab_reference': Path(slab_reference_path)
    }
    entry, species, reaction, slab = (_read(paths[key]) for key in (
        'catalog_entry',
        'species_recipe',
        'reaction_recipe',
        'slab_reference'
    ))
    metal = _validate_input(entry, species, reaction, slab)
    rid = entry['reaction_id']
    input_hashes = {key: _sha(path) for key, path in paths.items()}
    _save(
        output / 'input_provenance.json',
        {
            'reaction_id': rid,
            'recipe_sha256': input_hashes,
            'upstream_substrate_preparation_cost': slab.get('upstream_substrate_preparation_cost'),
            'upstream_cost_unavailable_reason': slab.get('upstream_cost_unavailable_reason'),
            'optimized_adsorbate_endpoint_reused': False
        }
    )
    begin_wall = time.monotonic()
    begin_cpu = time.process_time()
    events = []
    holder = {'factory': None}

    def check():
        if (output / 'STOP').exists() or any(((parent / 'STOP_NOW').exists() for parent in (output, *output.parents[:3]))):
            raise PreparationStopped('STOP_NOW_or_item_STOP')

    def timed(stage_id, fn, *, origin, inputs=()):
        current = holder['factory'] or SimpleNamespace(calls=0)
        clock = _clock(stage_id, rid, current, origin=origin, inputs=inputs)
        stage_dir = output / stage_id
        stage_dir.mkdir(exist_ok=False)
        try:
            check()
            value = fn(stage_dir)
            event = _finish_stage(
                stage_dir,
                clock,
                holder['factory'] or current,
                artifacts=(str(stage_dir),)
            )
            events.append(event)
            _save(output / 'stage_telemetry.json', events)
            return value
        except Exception as exc:
            event = _finish_stage(
                stage_dir,
                clock,
                holder['factory'] or current,
                status='interrupted' if isinstance(exc, PreparationStopped) else 'preparation_unconfirmed' if isinstance(exc, PreparationIncomplete) else 'preparation_error',
                reason=exc.reason_code if isinstance(exc, PreparationIncomplete) else f'{type(exc).__name__}: {exc}',
                artifacts=(str(stage_dir),)
            )
            events.append(event)
            _save(output / 'stage_telemetry.json', events)
            raise

    def embed(stage_dir):
        fragment = build_fragment(entry['species'], species['smiles'], species['mode'])
        if fragment.raw_symbols.count('Cl') != (len(fragment.holders) if species['mode'] == 'Cl' else 0) or (species['mode'] == 'Cl' and fragment.raw_symbols[fragment.binder] != species['binder_element']):
            raise ValueError('Cl placeholder/binder recipe mismatch')
        _save(
            stage_dir / 'embedded_fragment.json',
            {
                'raw_symbols': fragment.raw_symbols,
                'raw_positions': fragment.raw_positions,
                'binder_raw_index': fragment.binder,
                'holders_raw_indices': fragment.holders,
                'charge': fragment.charge,
                'spin': fragment.spin,
                'rdkit_steps': fragment.steps
            }
        )
        return fragment
    fragment = timed(
        'RDKit_embed',
        embed,
        origin='fresh_declared_molecular_SMILES',
        inputs=input_hashes.values()
    )

    def load(stage_dir):
        nonlocal factory_builder
        if factory_builder is None:
            from automation_search.uma_model import ModelFactory
            factory_builder = ModelFactory
        factory = factory_builder(check)
        holder['factory'] = factory
        _save(stage_dir / 'model_context.json', factory.metadata())
        return factory
    factory = timed('UMA_model_setup', load, origin='pinned_UMA_model_load')

    def gas(stage_dir):
        method = gas_relaxer or _gas_relax
        return method(fragment, factory, stage_dir, check)
    gas_diag = timed('OMOL_gas_parent', gas, origin='fresh_UMA_omol_with_removable_Cl')
    strategy = entry['endpoint_preparation_strategy']

    def place(stage_dir):
        finalize_direction(fragment)
        rows = []
        for index, variant in enumerate(_parent_variant_grid(entry, species, slab, fragment)):
            seed = _build_parent_seed(
                fragment,
                slab,
                entry,
                species,
                stage_dir,
                variant=variant,
                finalized=True,
                artifact_name=f'molecular_seed_{index + 1:02}.json'
            )
            rows.append((seed, variant))
        _save(
            stage_dir / 'seed_grid.json',
            {
                'candidates': [v for _, v in rows],
                'site_kind_count': len({v['site'] for _, v in rows})
            }
        )
        return rows
    parent_candidates = timed(
        'molecular_site_seed',
        place,
        origin='all_facet_site_kinds_times_declared_orientations'
    )
    prepared_parent = None

    def initial_new(stage_dir):
        selected = _select_parent_seed(
            parent_candidates,
            entry,
            factory,
            stage_dir,
            check,
            allow_none=strategy == 'REVERSE_BOND_SCAN',
            criterion_profile='v3'
        )
        if selected is None:
            _save(
                stage_dir / 'molecular_IS_validation.json',
                {
                    'valid_requested_bound_parent': False,
                    'fallback': 'existing_reverse_scan_then_calibrated_gas_boundary_if_needed',
                    'gas_fallback_is_not_bound_H2_success': True
                }
            )
            return None
        index, initial, diagnostic, seed, variant = selected
        role = diagnostic['selected_parent_role']
        _save(
            output / 'molecular_site_seed' / 'molecular_seed.json',
            _read(output / 'molecular_site_seed' / f'molecular_seed_{index + 1:02}.json')
        )
        _save(
            stage_dir / 'molecular_IS_validation.json',
            {
                **diagnostic,
                'valid_requested_molecular_graph': True,
                'selected_candidate_index': index,
                'selected_variant': variant,
                'valid_requested_parent_role': diagnostic['selected_parent_channel'] == 'requested_primary',
                'valid_requested_bound_parent': role in ('surface_contact', 'weak_contact')
            }
        )
        return (seed, variant, initial, diagnostic)
    selected_parent = timed(
        'OC20_IS_preparation' if strategy != 'REVERSE_BOND_SCAN' else 'OC20_parent_site_selection',
        initial_new,
        origin='lowest_energy_valid_multisite_bound_parent'
    )
    if selected_parent is not None:
        parent_seed, selected_variant, selected_is, selected_diag = selected_parent
        _apply_selected_site(entry, selected_variant)
        prepared_parent = (
            selected_is,
            selected_diag,
            'new_approved_parent_then_FS' if rid.startswith('N') else 'multisite_bound_parent_then_FS'
        )
    else:
        parent_seed, selected_variant = next(
            (
                (seed, variant) for seed,
                variant in parent_candidates if variant['site'] == entry['seed']['site'] and variant['orientation'] == entry['seed']['orientation']
            ),
            parent_candidates[0]
        )
        _apply_selected_site(entry, selected_variant)
    before_seed_hash = sha256(json.dumps(
        from_ase(parent_seed),
        sort_keys=True,
        allow_nan=False
    ).encode('utf-8')).hexdigest()

    def fs(stage_dir):
        return _candidate_fs(
            prepared_parent[0] if prepared_parent else parent_seed,
            entry,
            reaction,
            factory,
            stage_dir,
            check
        )
    anchor, moving, (final, fs_diag, selected_index) = timed(
        'OC20_FS_preparation',
        fs,
        origin='existing_route_FS_generator_and_relaxation'
    )

    def initial(stage_dir):
        if strategy == 'REVERSE_BOND_SCAN':
            result, diagnostic = _reverse_scan_is(
                final,
                parent_seed,
                anchor,
                moving,
                entry,
                factory,
                stage_dir,
                check
            )
            return (
                result,
                diagnostic,
                'existing_repeatability_calibrated_gas_boundary' if diagnostic.get('gas_boundary_verified') else 'existing_FS_reverse_bond_scan'
            )
        initial, diagnostic, valid = _aligned_is(
            final,
            parent_seed,
            anchor,
            moving,
            entry,
            factory,
            stage_dir,
            check
        )
        if valid:
            return (initial, diagnostic, 'existing_FS_aligned_molecular_relax')
        if strategy != 'REVERSE_BOND_SCAN':
            raise PreparationIncomplete('FS_aligned_IS_not_stable_requested_molecule')
        initial, diagnostic = _reverse_scan_is(
            final,
            parent_seed,
            anchor,
            moving,
            entry,
            factory,
            stage_dir,
            check
        )
        if diagnostic.get('gas_boundary_verified') is not True and (not diagnostic['converged'] or _edge_list(initial) != entry['expected_IS_edges']):
            raise PreparationIncomplete('scan_first_IS_not_valid_requested_molecule')
        return (
            initial,
            diagnostic,
            'existing_repeatability_calibrated_gas_boundary' if diagnostic.get('gas_boundary_verified') else 'existing_FS_reverse_bond_scan'
        )
    initial_atoms, is_diag, is_method = prepared_parent if prepared_parent else timed(
        'OC20_IS_preparation',
        initial,
        origin='existing_route_molecular_or_gas_endpoint'
    )

    def validate(stage_dir):
        if initial_atoms.get_chemical_symbols() != final.get_chemical_symbols() or not np.array_equal(
            initial_atoms.arrays['rw_id'],
            final.arrays['rw_id']
        ) or (not np.array_equal(initial_atoms.cell, final.cell)) or (not np.array_equal(initial_atoms.pbc, final.pbc)) or (assert_slab(initial_atoms) != assert_slab(final)):
            raise ValueError('Fresh endpoint composition/ID/cell/PBC/fixed-mask mismatch')
        if _edge_list(initial_atoms) != entry['expected_IS_edges'] or _edge_list(final) != entry['expected_FS_edges']:
            raise PreparationIncomplete('fresh_endpoint_target_graph_mismatch')
        pair = [
            int(initial_atoms.arrays['rw_id'][anchor]),
            int(initial_atoms.arrays['rw_id'][moving[0]])
        ]
        if pair != entry['target_pair_atom_ids']:
            raise ValueError('Fresh target atom ID mapping changed')
        r0 = float(initial_atoms.get_distance(anchor, moving[0], mic=True))
        dfs = float(final.get_distance(anchor, moving[0], mic=True))
        if not math.isfinite(r0) or not math.isfinite(dfs) or r0 <= 0 or (dfs <= r0):
            raise PreparationIncomplete('fresh_FS_target_bond_not_separated')
        _save(
            stage_dir / 'validated_pair.json',
            {
                'target_pair_atom_ids': pair,
                'r0_A': r0,
                'd_FS_A': dfs,
                'IS_edges': _edge_list(initial_atoms),
                'FS_edges': _edge_list(final),
                'IS_method': is_method,
                'FS_selected_candidate_index': selected_index,
                'Ni_substrate_upstream_cost_unknown': metal == 'Ni',
                'surface_element': metal,
                'same_atom_ID_cell_PBC_fixed_mask': True
            }
        )
        return (r0, dfs)
    r0, dfs = timed(
        'endpoint_pair_validation',
        validate,
        origin='fresh_same_composition_and_declared_graph_check'
    )
    config = deepcopy(entry['config'])
    config.setdefault('settings', {})['site'] = entry['settings']['site']
    config.setdefault('reaction', {}).update(id=rid, anchor=anchor, moving=moving[0])
    config.setdefault('endpoints', {}).update(IS_artifact='IS', FS_artifact='FS')
    config.setdefault('judge', {}).pop('ref_bond_A', None)
    record = {
        'artifacts': {'IS': from_ase(initial_atoms), 'FS': from_ase(final)},
        'IS': {key: is_diag.get(key) for key in (
            'converged',
            'steps',
            'fmax',
            'energy_eV'
        )},
        'FS': {key: fs_diag.get(key) for key in (
            'converged',
            'steps',
            'fmax',
            'energy_eV'
        )},
        'anchor': anchor,
        'moving': moving[0],
        'moving_fragment': moving,
        'expected_IS_edges': entry['expected_IS_edges'],
        'expected_FS_edges': entry['expected_FS_edges'],
        'settings': entry['settings'],
        'model': entry['model'],
        'task': entry['task'],
        'reaction': reaction,
        'preparation_method': is_method,
        'endpoint_preparation_strategy': strategy,
        'selected_IS_site': selected_variant['site'],
        'selected_parent_role': selected_diag.get('selected_parent_role') if selected_parent is not None else None,
        'selected_parent_channel': selected_diag.get('selected_parent_channel') if selected_parent is not None else 'reverse_scan_or_gas_boundary',
        'allowed_parent_roles': reaction_role_contract(rid)['allowed_parent_roles'],
        'endpoint_preparation_strategy_provenance': entry['endpoint_preparation_strategy_provenance'],
        'gas_preparation': {k: gas_diag[k] for k in (
            'converged',
            'steps',
            'e_before',
            'e_after'
        )}
    }
    elapsed = time.monotonic() - begin_wall
    cpu = time.process_time() - begin_cpu
    stage_sum = sum((row['wall_monotonic_s'] for row in events))
    total_cost = {
        'schema': 'cu_drm_fresh_endpoint_cost_v1' if metal == 'Cu' else 'ni_drm_fresh_endpoint_cost_v1',
        'whole_worker_wall_monotonic_s': elapsed,
        'whole_worker_process_cpu_s': cpu,
        'exclusive_stage_wall_sum_s': stage_sum,
        'unattributed_worker_overhead_s': max(0.0, elapsed - stage_sum),
        'model_setup_wall_s': next((row['wall_monotonic_s'] for row in events if row['stage_id'] == 'UMA_model_setup')),
        'model_load_inner_s': getattr(factory, 'load_seconds', None),
        'actual_predict_calls_total': getattr(factory, 'calls', None),
        'stage_model_calls_sum': sum((row['model_calls_delta'] or 0 for row in events)),
        'stages_nonoverlapping': True,
        'upstream_Ni_substrate_wall_cpu_calls': None,
        'upstream_cost_unavailable_reason': slab.get('upstream_cost_unavailable_reason')
    }
    if metal == 'Cu':
        total_cost.pop('upstream_Ni_substrate_wall_cpu_calls')
        total_cost['upstream_Cu_clean_slab_cost'] = slab.get('upstream_substrate_preparation_cost')
        total_cost['upstream_Cu_cost_accounting_role'] = 'shared_once_for_campaign_not_per_reaction'
    _save(output / 'preparation_cost.json', total_cost)
    if total_cost['stage_model_calls_sum'] != total_cost['actual_predict_calls_total']:
        raise ValueError('Fresh preparation stage model calls do not sum to factory total')
    prepared = {
        'reaction_id': rid,
        'species': entry['species'],
        'route': entry['route'],
        'requested_direction': entry['requested_direction'],
        'computational_direction': entry['computational_direction'],
        'target_pair_atom_ids': entry['target_pair_atom_ids'],
        'anchor': anchor,
        'moving': moving[0],
        'moving_fragment': moving,
        'expected_IS_edges': entry['expected_IS_edges'],
        'expected_FS_edges': entry['expected_FS_edges'],
        'is_structure': record['artifacts']['IS'],
        'endpoint_record': record,
        'config': config,
        'r0_A': r0,
        'r0_source': 'fresh prepared IS actual target pair MIC distance',
        'd_FS_A': dfs,
        'availability': AVAILABILITY,
        'input_path': str(output / 'prepared_entry.json'),
        'preparation_origin': 'fresh_molecule_from_SMILES_and_GUI_Cu111_clean_slab' if metal == 'Cu' else 'fresh_molecule_from_SMILES_and_new_Ni100_ase_slab' if slab.get('surface_profile', {}).get('facet') == 'fcc100' else 'fresh_molecule_from_SMILES_and_external_Ni_only_substrate',
        'endpoint_preparation_strategy': strategy,
        'selected_IS_site': selected_variant['site'],
        'selected_parent_role': record['selected_parent_role'],
        'selected_parent_channel': record['selected_parent_channel'],
        'allowed_parent_roles': record['allowed_parent_roles'],
        'endpoint_preparation_strategy_provenance': entry['endpoint_preparation_strategy_provenance'],
        'source_catalog_sha256_for_provenance_only': entry['old_source_sha256_for_provenance_only'],
        'fresh_recipe_sha256': input_hashes,
        'fresh_molecular_seed_geometry_sha256': before_seed_hash,
        'upstream_substrate_preparation_cost': slab.get('upstream_substrate_preparation_cost'),
        'upstream_cost_unavailable_reason': slab.get('upstream_cost_unavailable_reason'),
        'surface_element': metal,
        'surface_source_gui_job_id': slab.get('gui_surface_job_id') if metal == 'Cu' else None,
        'model_context': factory.metadata(),
        'preparation_cost': total_cost,
        'preparation_stage_telemetry_path': str(output / 'stage_telemetry.json')
    }
    _save(output / 'prepared_entry.json', prepared)
    return prepared
