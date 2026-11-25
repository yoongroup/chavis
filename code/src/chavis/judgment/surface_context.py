from __future__ import annotations
import math
import numpy as np
CU_CRITERION_VERSION_V1 = 'cu111-reaction-success-v1'
CU_CRITERION_VERSION_V2 = 'cu111-reaction-success-v2'
CU_CRITERION_VERSION = 'cu111-reaction-success-v3'
NI100_CRITERION_VERSION_V1 = 'ni100-reaction-success-v1'
NI100_CRITERION_VERSION_V2 = 'ni100-reaction-success-v2'
NI100_CRITERION_VERSION = 'ni100-reaction-success-v3'
RH111_CRITERION_VERSIONS = {
    'v1': 'rh111-reaction-success-v1',
    'v2': 'rh111-reaction-success-v2',
    'v3': 'rh111-reaction-success-v3'
}


def criterion_for(structure, *, criterion_profile='v2'):
    profile = (structure.get('meta') or {}).get('surface_profile') if isinstance(structure, dict) else None
    symbols = structure.get('symbols', []) if isinstance(structure, dict) else []
    if criterion_profile not in ('v1', 'v2', 'v3'):
        raise ValueError('UNSUPPORTED_JUDGMENT_PROFILE')
    if isinstance(profile, dict) and profile.get('element') == 'Rh' or 'Rh' in symbols:
        return RH111_CRITERION_VERSIONS[criterion_profile]
    if isinstance(profile, dict) and profile.get('element') == 'Ni' and (profile.get('facet') == 'fcc100'):
        return {
            'v1': NI100_CRITERION_VERSION_V1,
            'v2': NI100_CRITERION_VERSION_V2,
            'v3': NI100_CRITERION_VERSION
        }[criterion_profile]
    return {
        'v1': CU_CRITERION_VERSION_V1,
        'v2': CU_CRITERION_VERSION_V2,
        'v3': CU_CRITERION_VERSION
    }[criterion_profile] if isinstance(profile, dict) and profile.get('element') == 'Cu' or 'Cu' in symbols else None


def selected_surface_element(
    structure,
    *,
    positions=None,
    cell=None,
    symbols=None,
    adsorbate_indices=None,
    fixed_indices=None,
    pbc=None
):
    if not isinstance(structure, dict):
        raise ValueError('MISSING_STRUCTURE')
    symbols = list(structure['symbols'] if symbols is None else symbols)
    ads = list(structure.get('adsorbate_indices', []) if adsorbate_indices is None else adsorbate_indices)
    fixed = list(structure.get('fixed_indices', []) if fixed_indices is None else fixed_indices)
    pbc = list(structure.get('pbc', []) if pbc is None else pbc)
    cell = np.asarray(structure['cell'] if cell is None else cell, dtype=float)
    positions = np.asarray(
        structure['positions'] if positions is None else positions,
        dtype=float
    )
    slab = [index for index in range(len(symbols)) if index not in set(ads)]
    if len(slab) != 36 or len(fixed) != 18 or pbc != [True, True, False]:
        raise ValueError('UNSUPPORTED_SLAB_CONTEXT')
    metals = {symbols[index] for index in slab}
    if len(metals) != 1 or not metals <= {'Ni', 'Cu', 'Rh'}:
        raise ValueError('UNSUPPORTED_SLAB_ELEMENT')
    metal = metals.pop()
    if cell.shape != (3, 3) or positions.shape != (len(symbols), 3):
        raise ValueError('INVALID_SLAB_GEOMETRY')
    normal = np.cross(cell[0], cell[1])
    size = float(np.linalg.norm(normal))
    if not math.isfinite(size) or size <= 0:
        raise ValueError('INVALID_SLAB_NORMAL')
    normal /= size
    if normal[2] < 0:
        normal = -normal
    lower = set(sorted(slab, key=lambda index: float(positions[index] @ normal))[:18])
    if set(fixed) != lower:
        raise ValueError('FIXED_MASK_NOT_BOTTOM_18')
    profile = (structure.get('meta') or {}).get('surface_profile')
    if metal == 'Rh':
        if not isinstance(profile, dict):
            raise ValueError('RH_SAVED_SURFACE_PROFILE_MISSING')
        source = profile.get('source') or {}
        a = profile.get('a')
        saved_cell = np.asarray(profile.get('cell'), dtype=float)
        if (profile.get('element') != 'Rh' or profile.get('facet') != 'fcc111'
                or profile.get('size') != [3, 3, 4] or profile.get('n_slab') != 36
                or profile.get('fixed_layers') != 2
                or profile.get('fixed_layer_numbers') != [3, 4]
                or profile.get('fixed_indices') != fixed or profile.get('pbc') != pbc
                or source.get('kind') != 'ASE fcc111'
                or source.get('cell_relaxation') is not False
                or not isinstance(source.get('lattice_source'), str)
                or not source['lattice_source'].strip()
                or type(a) not in (int, float) or not math.isfinite(a) or a <= 0
                or profile.get('vacuum') != 10.0 or saved_cell.shape != (3, 3)
                or not np.allclose(saved_cell, cell, rtol=0, atol=1e-08)):
            raise ValueError('RH_APPROVED_FCC111_PROFILE_MISMATCH')
        from ase.build import fcc111
        expected = fcc111('Rh', size=(3, 3, 4), a=a, vacuum=10.0)
        if not np.allclose(expected.cell.array, cell, rtol=0, atol=1e-08):
            raise ValueError('RH_FCC111_CELL_LATTICE_MISMATCH')
        return metal
    if metal == 'Ni':
        if isinstance(profile, dict) and profile.get('element') != 'Ni':
            raise ValueError('SAVED_SURFACE_ELEMENT_MISMATCH')
        if profile is None:
            from ase.geometry import find_mic
            top = sorted(slab, key=lambda index: float(positions[index] @ normal))[-9:]
            neighbor_counts = []
            for index in top:
                others = [j for j in top if j != index]
                _, distances = find_mic(
                    positions[others] - positions[index],
                    cell,
                    pbc=pbc
                )
                spacing = float(np.min(distances))
                neighbor_counts.append(int(np.sum((distances > 0.87 * spacing) & (distances < 1.13 * spacing))))
            if any((count != 6 for count in neighbor_counts)):
                raise ValueError('UNPROFILED_NI_FACET_UNVERIFIED')
        if isinstance(profile, dict) and profile.get('facet') == 'fcc100':
            source = profile.get('source') or {}
            if profile.get('size') != [3, 3, 4] or profile.get('n_slab') != 36 or profile.get('fixed_layers') != 2 or (profile.get('fixed_indices') != fixed) or (profile.get('pbc') != pbc) or (source.get('kind') != 'ASE fcc100') or (profile.get('a') != 3.52) or (profile.get('vacuum') != 10.0) or (not isinstance(profile.get('cell'), list)) or (np.asarray(profile['cell']).shape != (3, 3)) or (not np.allclose(
                profile['cell'],
                cell,
                rtol=0,
                atol=1e-08
            )):
                raise ValueError('NI100_APPROVED_PROFILE_MISMATCH')
        elif isinstance(profile, dict) and profile.get('facet') != 'fcc111':
            raise ValueError('UNSUPPORTED_NI_SURFACE_PROFILE')
        return metal
    if not isinstance(profile, dict):
        raise ValueError('CU_SAVED_SURFACE_PROFILE_MISSING')
    source = profile.get('source') or {}
    if profile.get('element') != 'Cu' or profile.get('facet') != 'fcc111' or profile.get('size') != [3, 3, 4] or (profile.get('n_slab') != 36) or (profile.get('fixed_layers') != 2) or (profile.get('fixed_indices') != fixed) or (profile.get('pbc') != pbc) or (source.get('kind') != 'ASE fcc111') or (profile.get('a') != 3.61) or (profile.get('vacuum') != 10.0) or (not isinstance(profile.get('cell'), list)) or (np.asarray(profile['cell']).shape != (3, 3)) or (not np.allclose(
        profile['cell'],
        cell,
        rtol=0,
        atol=1e-08
    )):
        raise ValueError('CU_APPROVED_FCC111_PROFILE_MISMATCH')
    return metal
