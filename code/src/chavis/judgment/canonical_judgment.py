from __future__ import annotations
import math
import numpy as np
from .graph import RULES, structure_context as _structure
from .chemistry import classify_states, validate_input_reaction
from .policy import NUMERIC_POLICY, RADII, V1_PROFILE, V2_PROFILE, V3_PROFILE, H_H_PAIR_POLICY, bond_threshold, negative_curvature
from .surface_context import CU_CRITERION_VERSION, CU_CRITERION_VERSION_V1, CU_CRITERION_VERSION_V2, NI100_CRITERION_VERSION, NI100_CRITERION_VERSION_V1, NI100_CRITERION_VERSION_V2, RH111_CRITERION_VERSIONS, criterion_for, selected_surface_element
CRITERION_VERSION_V1 = 'ni-drm-reaction-success-v1'
CRITERION_VERSION_V2 = 'ni-drm-reaction-success-v2'
CRITERION_VERSION = 'ni-drm-reaction-success-v3'
TS_FMAX_EV_A = NUMERIC_POLICY['ts_fmax_eV_A']
ENDPOINT_FMAX_EV_A = NUMERIC_POLICY['endpoint_fmax_eV_A']
IMAGINARY_EPS_CM1 = NUMERIC_POLICY['imaginary_epsilon_cm1']


def descriptor_from_entry(entry, saved_IS):
    record = entry.get('endpoint_record') or {}
    reaction = record.get('reaction') or {}
    atom_ids = saved_IS.get('atom_ids')
    if not isinstance(atom_ids, list) or len(set(atom_ids)) != len(atom_ids):
        raise ValueError('Saved IS atom IDs are unavailable or nonunique')
    pair_ids = entry.get('target_pair_atom_ids')
    if pair_ids is None:
        pair = entry.get('target_pair')
        if pair is None:
            pair = [
                entry.get('anchor') if entry.get('anchor') is not None else record.get('anchor'),
                entry.get('moving') if entry.get('moving') is not None else record.get('moving')
            ]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any((type(index) is not int or index < 0 or index >= len(atom_ids) for index in pair)):
            raise ValueError('Saved target pair row indices are unavailable')
        pair_ids = [atom_ids[index] for index in pair]
    if not isinstance(pair_ids, (list, tuple)) or len(pair_ids) != 2 or any((type(atom_id) is not int or atom_id not in atom_ids for atom_id in pair_ids)) or (pair_ids[0] == pair_ids[1]):
        raise ValueError('Saved target pair atom IDs are unavailable')
    return {
        'target_pair_atom_ids': list(pair_ids),
        'target_bond_elements': entry.get('target_bond_elements') or reaction.get('broken_bond_elements'),
        'expected_IS_edges': entry.get('expected_IS_edges'),
        'expected_FS_edges': entry.get('expected_FS_edges'),
        'requested_direction': entry.get('requested_direction', 'dissociation'),
        'computational_direction': entry.get('computational_direction') or 'dissociation',
        'allowed_parent_roles': entry.get('allowed_parent_roles') or reaction.get('allowed_parent_roles'),
        'requested_formula': entry.get('requested_formula') or reaction.get('equation')
    }


def _finite(value):
    return isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(value)


def _isotope_context(structure):
    meta = structure.get('meta') or {}
    return structure.get(
        'isotopes',
        structure.get('atom_isotopes', meta.get('isotopes'))
    )


def same_context(reference, candidate):
    try:
        a = _structure(reference)
        b = _structure(candidate)
        if a[2:6] != b[2:6] or a[6] != b[6]:
            return (False, 'symbols_ids_adsorbate_fixed_or_pbc_changed')
        if not np.allclose(
            a[1],
            b[1],
            atol=NUMERIC_POLICY['cell_context_atol_A'],
            rtol=0
        ):
            return (False, 'cell_changed')
        if not np.allclose(
            a[0][a[5]],
            b[0][a[5]],
            atol=NUMERIC_POLICY['fixed_position_context_atol_A'],
            rtol=0
        ):
            return (False, 'fixed_slab_coordinates_changed')
        if len(a[5]) != 18 or len(a[0]) - len(a[4]) != 36 or set(a[5]) & set(a[4]):
            return (False, 'unsupported_ni36_bottom18_context')
        slab_element = selected_surface_element(
            reference,
            positions=a[0],
            cell=a[1],
            symbols=a[2],
            adsorbate_indices=a[4],
            fixed_indices=a[5],
            pbc=a[6]
        )
        profile = (reference.get('meta') or {}).get('surface_profile') or {}
        if slab_element in ('Cu', 'Rh') or profile.get('facet') == 'fcc100':
            candidate_element = selected_surface_element(
                candidate,
                positions=b[0],
                cell=b[1],
                symbols=b[2],
                adsorbate_indices=b[4],
                fixed_indices=b[5],
                pbc=b[6]
            )
            if candidate_element != slab_element or (reference.get('meta') or {}).get('surface_profile') != (candidate.get('meta') or {}).get('surface_profile'):
                return (
                    False,
                    'cu_surface_profile_changed' if slab_element == 'Cu' else 'saved_surface_profile_changed'
                )
        slab = [i for i in range(len(a[0])) if i not in a[4]]
        normal = np.cross(a[1][0], a[1][1])
        norm = float(np.linalg.norm(normal))
        if norm <= 0:
            return (False, 'invalid_slab_normal')
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        lower = set(sorted(slab, key=lambda i: float(a[0][i] @ normal))[:18])
        if set(a[5]) != lower:
            return (
                False,
                'fixed_mask_not_bottom_18_' + slab_element.lower()
            )
        if a[6] != [True, True, False]:
            return (False, 'unsupported_pbc')
        for key in ('initial_magmoms', 'initial_charges', 'tags', 'masses'):
            if key in reference or key in candidate:
                if reference.get(key) != candidate.get(key):
                    return (False, key + '_changed')
        for key in ('charge', 'spin'):
            if reference.get('meta', {}).get(key) != candidate.get('meta', {}).get(key):
                return (False, key + '_changed')
        if _isotope_context(reference) != _isotope_context(candidate):
            return (False, 'isotopes_changed')
        return (True, 'same_saved_model_context')
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return (False, 'invalid_saved_structure:' + str(exc))


def _normalize(raw):
    report = raw.get('report', raw)
    artifacts = raw.get('artifacts', report.get('artifacts', {}))
    ts = report.get('TS') if isinstance(report.get('TS'), dict) and 'positions' in report['TS'] else artifacts.get('TS')
    optimizer = report.get('dimer', report.get('ts', {}))
    frequency = report.get('frequencies', {})
    connections = report.get('connections', {})
    if isinstance(connections, dict) and 'results' in connections:
        connections = {row['direction']: {
            **row,
            'structure': artifacts.get('connection_' + row['direction'])
        } for row in connections['results'] if isinstance(row, dict) and row.get('direction') in ('minus', 'plus')}
    return (report, ts, optimizer, frequency, connections)


def spectrum_status(frequency, ts):
    if not isinstance(ts, dict):
        return {'status': 'unknown', 'reason': 'ts_missing', 'imaginary_count': None}
    fixed, symbols = (ts.get('fixed_indices'), ts.get('symbols'))
    if not isinstance(fixed, list) or not isinstance(symbols, list):
        return {
            'status': 'unknown',
            'reason': 'ts_mask_missing',
            'imaginary_count': None
        }
    free = [i for i in range(len(symbols)) if i not in fixed]
    values = frequency.get('cm1', frequency.get('cm-1')) if isinstance(frequency, dict) else None
    if not isinstance(frequency, dict) or frequency.get('completed', frequency.get('complete')) is not True or values is None:
        return {
            'status': 'unknown',
            'reason': 'complete_saved_frequency_missing',
            'imaginary_count': None
        }
    if not isinstance(values, list) or len(values) != 3 * len(free) or frequency.get('indices') != free or any((not _finite(value) for value in values)):
        return {
            'status': 'invalid',
            'reason': 'spectrum_not_finite_full_free_space',
            'imaginary_count': None
        }
    imaginary = [
        index for index,
        value in enumerate(values) if value < -IMAGINARY_EPS_CM1
    ]
    if frequency.get('imaginary_count') != len(imaginary) or frequency.get('imaginary_modes') != imaginary:
        return {
            'status': 'invalid',
            'reason': 'saved_imaginary_indices_or_count_inconsistent',
            'imaginary_count': len(imaginary)
        }
    return {
        'status': 'valid' if len(imaginary) == 1 else 'invalid',
        'reason': 'exactly_one_imaginary' if len(imaginary) == 1 else 'not_exactly_one_imaginary',
        'imaginary_count': len(imaginary),
        'free_atoms': len(free),
        'full_free_modes': len(values),
        'free_indices': free,
        'epsilon_cm1': IMAGINARY_EPS_CM1
    }


def physical_status(raw, saved_IS, ts, optimizer, frequency):
    spectral = spectrum_status(frequency, ts)
    if ts is None:
        return {'status': 'unknown', 'reason': 'ts_missing', 'spectrum': spectral}
    context_ok, context_reason = same_context(saved_IS, ts)
    if not context_ok:
        return {'status': 'invalid', 'reason': context_reason, 'spectrum': spectral}
    if spectral['status'] != 'valid':
        return {
            'status': spectral['status'],
            'reason': spectral['reason'],
            'spectrum': spectral
        }
    if not isinstance(optimizer, dict):
        optimizer = {}
    fmax, curvature = (optimizer.get('fmax'), optimizer.get('curvature'))
    if optimizer.get('converged') is not True or not _finite(fmax) or (not _finite(curvature)):
        explicit = optimizer.get('converged') is False
        return {
            'status': 'invalid' if explicit else 'unknown',
            'reason': 'ts_explicitly_nonconverged' if explicit else 'physical_ts_force_or_curvature_missing',
            'spectrum': spectral
        }
    if not 0 <= fmax <= TS_FMAX_EV_A or not negative_curvature(curvature):
        return {
            'status': 'invalid',
            'reason': 'physical_force_or_curvature_failed',
            'fmax_eV_A': fmax,
            'curvature_eV_A2': curvature,
            'spectrum': spectral
        }
    report = raw.get('report', raw)
    judge = raw.get('ts_judge', report.get('ts_judge'))
    if judge not in (None, 'OK'):
        return {
            'status': 'invalid',
            'reason': 'saved_ts_geometry_judge_failed',
            'ts_judge': judge,
            'spectrum': spectral
        }
    return {
        'status': 'valid',
        'reason': 'saved_first_order_stationary_candidate',
        'fmax_eV_A': fmax,
        'curvature_eV_A2': curvature,
        'fmax_limit_eV_A': TS_FMAX_EV_A,
        'spectrum': spectral,
        'saved_ts_judge': judge
    }


def downhill_status(connections, saved_IS):
    branches = {}
    for direction in ('minus', 'plus'):
        row = connections.get(direction) if isinstance(connections, dict) else None
        if not isinstance(row, dict) or not isinstance(row.get('structure'), dict):
            branches[direction] = {
                'status': 'unknown',
                'reason': 'saved_arrival_missing'
            }
            continue
        context_ok, context_reason = same_context(saved_IS, row['structure'])
        if not context_ok:
            branches[direction] = {'status': 'invalid', 'reason': context_reason}
            continue
        fmax = row.get('fmax')
        if row.get('converged') is False or (_finite(fmax) and (not 0 <= fmax <= ENDPOINT_FMAX_EV_A)):
            branches[direction] = {
                'status': 'invalid',
                'reason': 'saved_downhill_nonconverged',
                'fmax_eV_A': fmax
            }
        elif row.get('converged') is not True or not _finite(fmax):
            branches[direction] = {
                'status': 'unknown',
                'reason': 'saved_downhill_force_missing'
            }
        elif row.get('judge') not in (None, 'OK'):
            branches[direction] = {
                'status': 'invalid',
                'reason': 'saved_downhill_geometry_judge_failed',
                'judge': row['judge']
            }
        else:
            branches[direction] = {
                'status': 'valid',
                'reason': 'saved_downhill_converged',
                'fmax_eV_A': fmax
            }
    status = 'valid' if all((row['status'] == 'valid' for row in branches.values())) else 'invalid' if any((row['status'] == 'invalid' for row in branches.values())) else 'unknown'
    return {
        'status': status,
        'branches': branches,
        'fmax_limit_eV_A': ENDPOINT_FMAX_EV_A
    }


def _strict_auxiliary(sidecar):
    if isinstance(sidecar, dict) and isinstance(sidecar.get('strict_is_fs'), dict):
        sidecar = sidecar['strict_is_fs']
    if sidecar is True or (isinstance(sidecar, dict) and sidecar.get('status') == 'confirmed'):
        return (
            True,
            {'status': 'confirmed', 'source': 'saved_original_strict_sidecar'}
        )
    if sidecar is False or (isinstance(sidecar, dict) and sidecar.get('status') == 'unconfirmed'):
        return (
            False,
            {
                'status': 'unconfirmed',
                'source': 'saved_original_strict_sidecar',
                'reason': sidecar.get('reason') if isinstance(sidecar, dict) else None
            }
        )
    return (
        None,
        {'status': 'unknown', 'reason': 'original_strict_sidecar_not_provided'}
    )


def _energy_review(raw, report, optimizer, connections, chemistry, requested_direction):
    ts_energy = optimizer.get('energy_eV') if isinstance(optimizer, dict) else None
    basis = 'same_candidate_saved_ts_energy_eV'
    if not _finite(ts_energy) and _finite(raw.get('E_TS_minus_IS_eV')) and _finite(raw.get('reference_IS_energy_eV')):
        ts_energy = raw['reference_IS_energy_eV'] + raw['E_TS_minus_IS_eV']
        basis = 'saved_reference_IS_plus_saved_TS_gap'
    if not _finite(ts_energy):
        ts_energy = None
        basis = None
    arrivals = {direction: connections.get(direction, {}).get('energy_eV') for direction in ('minus', 'plus')}
    gaps = {
        direction: float(ts_energy - value) if ts_energy is not None and _finite(value) else None for direction,
        value in arrivals.items()
    }
    negative = [
        direction for direction,
        value in gaps.items() if value is not None and value < 0
    ]
    orientation = chemistry.get('orientation') if isinstance(chemistry, dict) else None
    start_role = 'products' if requested_direction == 'association' else 'parent'
    forward_start = next(
        (direction for direction, role in orientation.items() if role == start_role),
        None
    ) if isinstance(orientation, dict) else None
    return {
        'TS_eV': ts_energy,
        'actual_arrival_energy_eV': arrivals,
        'TS_minus_actual_arrival_eV': gaps,
        'energy_basis': basis,
        'negative_gap_branches_needing_review': negative,
        'energy_consistency_tolerance_eV': None,
        'requested_direction': requested_direction,
        'actual_connected_forward_start_branch': forward_start,
        'actual_connected_forward_barrier_eV': gaps.get(forward_start) if forward_start else None,
        'vetoes_primary_success': False
    }


def evaluate(
    result,
    saved_IS,
    saved_FS,
    *,
    species,
    reaction_id,
    input_descriptor,
    requested_direction='dissociation',
    strict_sidecar=None,
    mode_metrics=None,
    primary_candidate_identifier=None,
    criterion_profile=V2_PROFILE
):
    if not isinstance(result, dict):
        result = {}
    if criterion_profile not in (V1_PROFILE, V2_PROFILE, V3_PROFILE):
        raise ValueError('UNSUPPORTED_JUDGMENT_PROFILE')
    report, ts, optimizer, frequency, connections = _normalize(result)
    input_check = validate_input_reaction(
        saved_IS,
        saved_FS,
        species=species,
        descriptor=input_descriptor,
        criterion_profile=criterion_profile
    )
    physical = physical_status(result, saved_IS, ts, optimizer, frequency)
    downhill = downhill_status(connections, saved_IS)
    chemistry = {'status': 'unknown', 'reason': 'downhill_evidence_not_both_valid'}
    if input_check['status'] == 'validated' and downhill['status'] == 'valid':
        chemistry = classify_states(
            saved_IS,
            saved_FS,
            connections['minus']['structure'],
            connections['plus']['structure'],
            species=species,
            reaction_id=reaction_id,
            target_pair_atom_ids=input_descriptor['target_pair_atom_ids'],
            requested_direction=requested_direction,
            input_reaction=input_descriptor,
            criterion_profile=criterion_profile
        )
    strict_bool, strict_detail = _strict_auxiliary(strict_sidecar)
    energy = _energy_review(
        result,
        report,
        optimizer,
        connections,
        chemistry,
        requested_direction
    )
    gate_status = {
        'input_graph': input_check['status'],
        'physical_ts': physical['status'],
        'downhill': downhill['status'],
        'endpoint_chemistry': chemistry['status']
    }
    success = input_check['status'] == 'validated' and physical['status'] == 'valid' and (downhill['status'] == 'valid') and (chemistry['status'] == 'compatible')
    if input_check['status'] == 'input_reaction_mismatch':
        status = 'input_invalid'
    elif success:
        status = 'success'
    elif physical['status'] == 'invalid' or downhill['status'] == 'invalid' or chemistry['status'] == 'incompatible':
        status = 'unconfirmed'
    else:
        status = 'evidence_missing'
    reasons = []
    if input_check['status'] != 'validated':
        reasons.append('INPUT_REACTION_' + input_check['status'].upper())
    if physical['status'] != 'valid':
        reasons.append('PHYSICAL_TS_' + physical['reason'].upper())
    if downhill['status'] != 'valid':
        reasons.append('DOWNHILL_' + downhill['status'].upper())
    if chemistry['status'] != 'compatible':
        reasons.append('ENDPOINT_CHEMISTRY_' + chemistry['reason'].upper())
    if success:
        reasons = ['SAME_CANDIDATE_PHYSICAL_TS_AND_TARGET_ENDPOINT_CHEMISTRY']
    if energy['negative_gap_branches_needing_review']:
        reasons.append('NEGATIVE_SAVED_TS_MINUS_ARRIVAL_ENERGY_REVIEW_ONLY')
    criterion_version = criterion_for(
        saved_IS,
        criterion_profile=criterion_profile
    ) or {
        V1_PROFILE: CRITERION_VERSION_V1,
        V2_PROFILE: CRITERION_VERSION_V2,
        V3_PROFILE: CRITERION_VERSION
    }[criterion_profile]
    response = {
        'criterion_version': criterion_version,
        'status': status,
        'success': bool(success),
        'primary_success': bool(success),
        'reason_codes': reasons,
        'gate_status': gate_status,
        'evidence_basis': 'mode_displaced_endpoint_chemistry',
        'formal_irc_verified': False,
        'trajectory_evidence_unavailable': True,
        'primary_candidate_identifier': primary_candidate_identifier,
        'input_graph_gate': input_check,
        'physical_ts': physical,
        'raw_one_imaginary_count': physical.get('spectrum', {}).get('imaginary_count'),
        'downhill': downhill,
        'endpoint_chemistry_gate': chemistry,
        'strict_match': strict_bool,
        'strict_match_detail': strict_detail,
        'mode_participation': mode_metrics if isinstance(mode_metrics, dict) else {
            'status': 'unknown',
            'reason': 'saved_mode_metric_not_provided'
        },
        'channel_variant': chemistry.get('channel_subtype'),
        'site_channel': chemistry.get('site_channel'),
        'actual_mapping': chemistry.get('graph_evidence', {}).get('consistent_mapping_actual_to_reference_atom_ids'),
        'requested_direction': requested_direction,
        'energy_review': energy,
        'review_flags': {
            'negative_actual_arrival_energy_gaps': energy['negative_gap_branches_needing_review'],
            'frozen_energy_tolerance_eV': None,
            'missing_mode_participation_non_veto': not isinstance(mode_metrics, dict) or mode_metrics.get('status') != 'available',
            'trajectory_evidence_unavailable_non_veto': True
        },
        'threshold_provenance': {
            'TS_fmax_eV_A': TS_FMAX_EV_A,
            'downhill_fmax_eV_A': ENDPOINT_FMAX_EV_A,
            'imaginary_epsilon_cm1': IMAGINARY_EPS_CM1,
            'reference_bond_intact_ratio': RULES['reference_bond_intact_ratio'],
            'nonreference_bond_radius_scale': RULES['nonreference_bond_radius_scale'],
            'near_Ni_radius_scale': RULES['near_Ni_radius_scale'],
            'detached_Ni_radius_scale': RULES['detached_Ni_radius_scale'],
            'detached_min_height_A': RULES['detached_min_height_A']
        }
    }
    if criterion_profile in (V2_PROFILE, V3_PROFILE):
        response['threshold_provenance'].update(
            h_h_pair_policy_key=H_H_PAIR_POLICY['key'],
            h_h_relaxed_gas_reference_A=H_H_PAIR_POLICY['reference_length_A'],
            h_h_intact_cutoff_A=bond_threshold('H', 'H'),
            h_h_model_task=H_H_PAIR_POLICY['model_task'],
            h_h_source=H_H_PAIR_POLICY['source']
        )
        weak = bool(success and chemistry.get('weak_contact_parent'))
        response['primary_success_label'] = 'primary_success_weak_contact' if weak else 'primary_success' if success else None
        response['review_flags']['weak_contact_parent'] = weak
    if criterion_profile == V3_PROFILE:
        response['requested_role_contract'] = chemistry.get('requested_role_contract')
        response['requested_role_compatible'] = chemistry.get('requested_role_compatible')
        response['nonprimary_channel_label'] = 'requested_roles_mismatch' if chemistry.get('bond_edit_compatible') and chemistry.get('requested_role_compatible') is False else None
    if criterion_version in (
        CU_CRITERION_VERSION,
        CU_CRITERION_VERSION_V2,
        CU_CRITERION_VERSION_V1
    ):
        response['surface_context'] = {
            'element': 'Cu',
            'facet': 'fcc111',
            'slab_atoms': 36,
            'fixed_bottom_atoms': 18,
            'source': 'saved_approved_GUI_surface_profile'
        }
        response['threshold_provenance'].update(
            surface_metal_radius_A=RADII['Cu'],
            contact_ratio_rule='same_named_numeric_policy_with_Cu_covalent_radius'
        )
    elif criterion_version in (
        NI100_CRITERION_VERSION,
        NI100_CRITERION_VERSION_V2,
        NI100_CRITERION_VERSION_V1
    ):
        response['surface_context'] = {
            'element': 'Ni',
            'facet': 'fcc100',
            'slab_atoms': 36,
            'fixed_bottom_atoms': 18,
            'source': 'saved_approved_ASE_fcc100_surface_profile'
        }
    elif criterion_version in RH111_CRITERION_VERSIONS.values():
        response['surface_context'] = {
            'element': 'Rh',
            'facet': 'fcc111',
            'slab_atoms': 36,
            'fixed_bottom_atoms': 18,
            'source': 'saved_approved_ASE_fcc111_surface_profile'
        }
        response['threshold_provenance'].update(
            surface_metal_radius_A=RADII['Rh'],
            contact_ratio_rule='same_named_numeric_policy_with_Rh_covalent_radius'
        )
    return response
