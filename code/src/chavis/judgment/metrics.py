from __future__ import annotations
import math
import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.geometry import find_mic
PROVENANCE = {
    'ase_vibrations_get_mode_cartesian_full',
    'ase_vibrations_get_mode_cartesian_full_rescaled'
}


def _unknown(
    reason,
    *,
    status='unknown',
    metadata=None,
    input_pair=None,
    scored_pair=None
):
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        'status': status,
        'reason': reason,
        'P_ads': None,
        'P_bond': None,
        'P_bond_given_ads': None,
        'percent_ads': None,
        'percent_bond': None,
        'target_pair_input_atom_ids': input_pair,
        'target_pair_scored_atom_ids': scored_pair,
        'target_pair_basis': 'mapped_actual_reactive_pair' if scored_pair is not None and input_pair != scored_pair else 'original_input_pair',
        'mode_provenance': {key: metadata.get(key) for key in (
            'provenance_token',
            'source_type',
            'source_path',
            'source_hash',
            'imaginary_index'
        )},
        'mass_provenance': metadata.get('mass_provenance'),
        'masses_amu': None,
        'free_indices': None,
        'atom_ids': None,
        'warnings': []
    }


def _masses(ts, metadata, symbols):
    provenance = metadata.get('mass_provenance')
    if provenance == 'vibrations_atoms_get_masses':
        value = metadata.get('masses_amu')
        if value is None:
            value = ts.get('masses')
        if value is None:
            raise ValueError('EXPLICIT_VIBRATION_MASSES_MISSING')
        masses = np.asarray(value, float)
        if ts.get('masses') is not None and (not np.allclose(
            masses,
            np.asarray(ts['masses'], float),
            rtol=0,
            atol=1e-10
        )):
            raise ValueError('SAVED_AND_VIBRATION_MASSES_DISAGREE')
    elif provenance == 'ase_default_masses_from_frozen_loader':
        if ts.get('masses') is not None:
            raise ValueError('EXPLICIT_SAVED_MASSES_WITH_DEFAULT_MASS_PROVENANCE')
        masses = np.asarray(
            [atomic_masses[atomic_numbers[symbol]] for symbol in symbols],
            float
        )
    else:
        raise ValueError('VIBRATION_MASS_PROVENANCE_UNVERIFIED')
    if masses.shape != (len(symbols),) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('INVALID_VIBRATION_MASSES')
    return (masses, provenance)


def score_saved_mode(
    IS,
    TS,
    mode,
    *,
    target_pair_atom_ids,
    mapped_target_pair_atom_ids=None,
    mode_metadata=None
):
    meta = mode_metadata if isinstance(mode_metadata, dict) else {}
    input_pair = list(target_pair_atom_ids) if isinstance(target_pair_atom_ids, (tuple, list)) else None
    scored_pair = list(mapped_target_pair_atom_ids) if isinstance(
        mapped_target_pair_atom_ids,
        (tuple, list)
    ) else input_pair
    if not isinstance(IS, dict) or not isinstance(TS, dict) or mode is None:
        return _unknown(
            'SAVED_STRUCTURE_OR_MODE_MISSING',
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
    if meta.get('provenance_token') not in PROVENANCE:
        return _unknown(
            'CARTESIAN_MODE_PROVENANCE_UNVERIFIED',
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
    source_type = meta.get('source_type')
    if source_type == 'direct_artifacts_modes' and meta['provenance_token'] != 'ase_vibrations_get_mode_cartesian_full' or (source_type == 'scan_imaginary_mode_npy' and meta['provenance_token'] != 'ase_vibrations_get_mode_cartesian_full_rescaled') or source_type not in (
        'direct_artifacts_modes',
        'scan_imaginary_mode_npy'
    ):
        return _unknown(
            'MODE_SOURCE_AND_CARTESIAN_PROVENANCE_DISAGREE',
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
    freq = meta.get('frequencies')
    if not isinstance(freq, dict):
        return _unknown(
            'SAVED_FREQUENCY_METADATA_MISSING',
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
    if freq.get('complete', freq.get('completed')) is not True:
        return _unknown(
            'SAVED_FREQUENCY_METADATA_INCOMPLETE',
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
    imaginary = freq.get('imaginary_modes')
    count = freq.get('imaginary_count')
    selected = meta.get('imaginary_index')
    if not isinstance(imaginary, list) or type(count) is not int or (not all((type(i) is int for i in imaginary))) or (count != len(imaginary)):
        return _unknown(
            'IMAGINARY_MODE_COUNT_UNVERIFIED',
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
    if count != 1 and (not meta.get('selected_mode_explicitly')):
        return _unknown(
            'NOT_EXACTLY_ONE_IMAGINARY_MODE',
            status='not_available_for_one_imag_score',
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
    if type(selected) is not int or selected not in imaginary:
        return _unknown(
            'SELECTED_MODE_NOT_SAVED_IMAGINARY_INDEX',
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
    try:
        symbols = TS['symbols']
        ids = TS['atom_ids']
        positions = np.asarray(TS['positions'], float)
        if not isinstance(symbols, list) or not isinstance(ids, list) or len(set(ids)) != len(ids) or (len(symbols) != len(ids)) or (positions.shape != (len(ids), 3)) or (not np.isfinite(positions).all()):
            raise ValueError('INVALID_SAVED_TS_ATOM_CONTEXT')
        if IS.get('atom_ids') != ids or IS.get('symbols') != symbols:
            raise ValueError('MODE_TS_AND_SAVED_IS_ATOM_IDS_DIFFER')
        if meta.get('atom_ids') != ids:
            raise ValueError('MODE_METADATA_ATOM_ID_ALIGNMENT_UNVERIFIED')
        fixed = TS.get('fixed_indices')
        ads = TS.get('adsorbate_indices', TS.get('meta', {}).get('adsorbate_indices'))
        if not isinstance(fixed, list) or not isinstance(ads, list) or any((type(i) is not int or i < 0 or i >= len(ids) for i in fixed + ads)) or (len(set(fixed)) != len(fixed)) or (len(set(ads)) != len(ads)):
            raise ValueError('INVALID_FIXED_OR_ADSORBATE_INDICES')
        free = [i for i in range(len(ids)) if i not in fixed]
        if freq.get('indices') != free or meta.get('free_indices') != free:
            raise ValueError('MODE_FREE_INDEX_ALIGNMENT_UNVERIFIED')
        signed = freq.get('cm-1', freq.get('cm1'))
        if not isinstance(signed, list) or len(signed) != 3 * len(free) or selected < 0 or (selected >= len(signed)):
            raise ValueError('SAVED_FREQUENCY_MODE_DIMENSION_MISMATCH')
        u = np.asarray(mode, float)
        if u.shape != (len(ids), 3) or not np.isfinite(u).all():
            raise ValueError('INVALID_FULL_CARTESIAN_MODE_ARRAY')
        mode_norm2 = float(np.sum(u * u))
        if mode_norm2 <= 0 or float(np.sum(u[fixed] ** 2)) > 1e-16 * mode_norm2:
            raise ValueError('MODE_FIXED_ENTRIES_NOT_ZERO')
        masses, mass_source = _masses(TS, meta, symbols)
        if not isinstance(scored_pair, list) or len(scored_pair) != 2 or any((type(atom_id) is not int or atom_id not in ids for atom_id in scored_pair)):
            raise ValueError('INVALID_SCORED_TARGET_PAIR_ATOM_IDS')
        by_id = {atom_id: index for index, atom_id in enumerate(ids)}
        a, b = (by_id[atom_id] for atom_id in scored_pair)
        if a == b or a not in ads or b not in ads:
            raise ValueError('SCORED_TARGET_PAIR_NOT_TWO_ADSORBATE_ATOMS')
        vector, distance = find_mic(
            positions[b] - positions[a],
            np.asarray(TS['cell'], float),
            pbc=TS['pbc']
        )
        distance = float(distance)
        if not math.isfinite(distance) or distance <= 1e-12:
            raise ValueError('UNDEFINED_MIC_TARGET_BOND_VECTOR')
        unit = np.asarray(vector, float) / distance
        S = float(np.sum(masses[free, None] * u[free] ** 2))
        if not math.isfinite(S) or S <= 0:
            raise ValueError('ZERO_OR_NONFINITE_MASS_WEIGHTED_MODE_NORM')
        ads_free = sorted(set(ads).intersection(free))
        p_ads = float(np.sum(masses[ads_free, None] * u[ads_free] ** 2) / S)
        reduced_mass = float(masses[a] * masses[b] / (masses[a] + masses[b]))
        stretch = float(unit @ (u[b] - u[a]))
        p_bond = float(reduced_mass * stretch * stretch / S)
        if not math.isfinite(p_ads) or not math.isfinite(p_bond) or p_ads < -1e-12 or (p_ads > 1 + 1e-10) or (p_bond < -1e-12) or (p_bond > 1 + 1e-10):
            raise ValueError('PARTICIPATION_OUTSIDE_PHYSICAL_UNIT_INTERVAL')
        p_ads = min(1.0, max(0.0, p_ads))
        p_bond = min(1.0, max(0.0, p_bond))
        return {
            'status': 'available',
            'reason': 'EXPLICITLY_SELECTED_MULTIIMAG_MODE_SUPPORT_ONLY' if count != 1 else 'SAVED_ONE_IMAGINARY_CARTESIAN_MODE_SCORED',
            'P_ads': p_ads,
            'P_bond': p_bond,
            'P_bond_given_ads': p_bond / p_ads if p_ads > 0 else None,
            'percent_ads': 100 * p_ads,
            'percent_bond': 100 * p_bond,
            'target_pair_input_atom_ids': input_pair,
            'target_pair_scored_atom_ids': scored_pair,
            'target_pair_basis': 'mapped_actual_reactive_pair' if input_pair != scored_pair else 'original_input_pair',
            'mode_provenance': {key: meta.get(key) for key in (
                'provenance_token',
                'source_type',
                'source_path',
                'source_hash',
                'imaginary_index'
            )},
            'mass_provenance': mass_source,
            'masses_amu': masses.tolist(),
            'free_indices': free,
            'atom_ids': ids,
            'normalization': 'mass_weighted_total_free_mode_norm',
            'warnings': [
                'ADSORBATE_MOTION_IS_NOT_EQUIVALENT_TO_TARGET_BOND_STRETCH',
                'PARTICIPATION_IS_SUPPORTING_EVIDENCE_NOT_A_TS_PASS_CUTOFF'
            ]
        }
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return _unknown(
            str(exc),
            metadata=meta,
            input_pair=input_pair,
            scored_pair=scored_pair
        )
