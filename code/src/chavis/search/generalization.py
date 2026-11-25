import math
REACTIONS = {
    'D01': dict(system='CH4', anchor=36, moving=37, config='jobs_ch4_v2.json'),
    'D05': dict(system='CO2', anchor=37, moving=36, config='jobs_D05_v2.json'),
    'D08': dict(system='H2O', anchor=36, moving=37, config='jobs_D08_v2.json'),
    'D13': dict(system='H2', anchor=36, moving=37, config='jobs_D13_v2.json')
}
PILOT_IMAGES = 17
PILOT_DIMER_CAP = 2
PILOT_SCIENCE_SECONDS = 300
PLACEMENT_DISTANCES_A = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def _numbers(values, label):
    if not isinstance(values, (list, tuple)) or not values or any((type(x) not in (int, float) or not math.isfinite(x) for x in values)):
        raise ValueError(f'{label} requires finite numbers')
    return list(values)


def _median_bond_index(indices, bonds):
    if not indices:
        return None
    median = sorted((bonds[i] for i in indices))[(len(indices) - 1) // 2]
    return min(indices, key=lambda i: (abs(bonds[i] - median), i))


def band_cohorts(energies, bonds, r0_A, *, peak_window_eV=0.5):
    energies, bonds = (_numbers(energies, 'energies'), _numbers(bonds, 'bonds'))
    if len(energies) != len(bonds) or not 3 <= len(bonds) <= 40 or min(bonds) <= 0:
        raise ValueError('Matched positive 3..40-image IDPP band required')
    if type(r0_A) not in (int, float) or not math.isfinite(r0_A) or r0_A <= 0:
        raise ValueError('Positive original saved-IS r0 required')
    if type(peak_window_eV) not in (int, float) or not math.isfinite(peak_window_eV) or peak_window_eV < 0:
        raise ValueError('Finite nonnegative peak window required')
    peak = max(range(1, len(energies) - 1), key=lambda i: (energies[i], -i))
    contiguous = {peak}
    for direction in (-1, 1):
        i = peak + direction
        while 0 < i < len(energies) - 1 and energies[peak] - energies[i] <= peak_window_eV:
            contiguous.add(i)
            i += direction
    geometry = {i for i in range(1, len(bonds) - 1) if 1.2 * r0_A <= bonds[i] <= 2.0 * r0_A}
    return dict(
        peak=peak,
        peak_cohort=sorted(contiguous),
        geometry_cohort=sorted(geometry),
        eligible_union=sorted(contiguous | geometry),
        original_saved_IS_r0_A=float(r0_A),
        target_window_A=[1.2 * r0_A, 2.0 * r0_A]
    )


def pilot_method1_seeds(bands, r0_A):
    if not isinstance(bands, dict) or set(bands) != {'rigid', 'relax'}:
        raise ValueError('Pilot requires rigid and relaxed bands')
    cohorts = {
        name: band_cohorts(row['energies'], row['bonds'], r0_A) for name,
        row in bands.items()
    }
    for name, row in bands.items():
        if len(row['bonds']) != PILOT_IMAGES:
            raise ValueError(f'{name} pilot band must have exactly 17 images')
    selected = []
    for cohort, preferred, fallback in (
        ('peak_cohort', 'rigid', 'relax'),
        ('geometry_cohort', 'relax', 'rigid')
    ):
        choice = None
        for band in (preferred, fallback):
            available = [i for i in cohorts[band][cohort] if not any((x['band'] == band and x['index'] == i for x in selected))]
            if available:
                choice = dict(
                    band=band,
                    index=_median_bond_index(available, bands[band]['bonds']),
                    cohort='peak' if cohort == 'peak_cohort' else 'geometry',
                    selection_reason='median_bond_in_reserved_cohort_tie_lower_index'
                )
                choice['eligible_cohorts'] = [
                    kind for kind,
                    key in (
                        ('peak', 'peak_cohort'),
                        ('geometry', 'geometry_cohort')
                    ) if choice['index'] in cohorts[band][key]
                ]
                break
        if choice is not None:
            selected.append(choice)
    return dict(
        cohorts=cohorts,
        seeds=selected,
        cap=PILOT_DIMER_CAP,
        original_saved_IS_r0_A=float(r0_A),
        geometry_coverage=any(('geometry' in x['eligible_cohorts'] for x in selected)),
        distinct_geometry_seed_available=any((x['cohort'] == 'geometry' for x in selected)),
        missing_cohorts=[name for name in ('peak', 'geometry') if not any((name in x['eligible_cohorts'] for x in selected))]
    )


def pilot_method2_seeds(energies, bonds, r0_A):
    energies, bonds = (_numbers(energies, 'energies'), _numbers(bonds, 'bonds'))
    if len(energies) != len(bonds) or min(bonds) <= 0 or type(r0_A) not in (int, float) or (not math.isfinite(r0_A)) or (r0_A <= 0):
        raise ValueError('Invalid scan and saved-IS r0')
    peak = max(range(len(energies)), key=lambda i: (energies[i], -i))
    stage_a = sorted(sorted(
        range(len(energies)),
        key=lambda i: (abs(i - peak), i)
    )[:10])
    geometry = [i for i, r in enumerate(bonds) if 1.2 * r0_A <= r <= 2.0 * r0_A]
    residual_b = [i for i in geometry if i not in stage_a]
    distinct_geometry = residual_b or [i for i in geometry if i != peak]
    seeds = [dict(
        index=peak,
        cohort='peak',
        selection_reason='scan_energy_peak_tie_lower_index'
    )]
    if distinct_geometry:
        seeds.append(dict(
            index=_median_bond_index(distinct_geometry, bonds),
            cohort='geometry',
            selection_reason='median_bond_residual_B_else_nonpeak_A_window_tie_lower_index'
        ))
    for row in seeds:
        row['eligible_cohorts'] = ['peak'] + (['geometry'] if row['index'] in geometry else []) if row['cohort'] == 'peak' else ['geometry'] + (['historical_stage_a'] if row['index'] in stage_a else [])
    return dict(
        peak=peak,
        historical_stage_a=stage_a,
        geometry_cohort=geometry,
        historical_residual_b_cohort=residual_b,
        seeds=seeds,
        cap=PILOT_DIMER_CAP,
        geometry_coverage=any((row['index'] in geometry for row in seeds)),
        distinct_geometry_seed_available=bool(distinct_geometry),
        missing_cohorts=[] if geometry else ['geometry'],
        original_saved_IS_r0_A=float(r0_A),
        target_window_A=[1.2 * r0_A, 2.0 * r0_A]
    )


def choose_near_placement(saved_is, saved_fs, anchor):
    import numpy as np
    from ase.geometry import find_mic
    from .structures import assert_compatible, dump
    from .surface_sites import _collisions
    assert_compatible(saved_is, saved_fs)
    ads = list(saved_is.info['adsorbate_indices'])
    if anchor not in ads:
        raise ValueError('Anchor must be in labeled adsorbate')
    ni = [i for i, atom in enumerate(saved_is) if atom.symbol == 'Ni']
    z0 = float(np.sort(saved_is.positions[ni, 2])[-9:].mean())
    vector, length = find_mic(
        saved_is.positions[anchor] - saved_fs.positions[anchor],
        saved_is.cell,
        saved_is.pbc
    )
    if length <= 1e-12:
        raise ValueError('Undefined saved anchor separation direction')
    options = []
    original_links = find_mic(
        saved_is.positions[ads] - saved_fs.positions[ads],
        saved_is.cell,
        saved_is.pbc
    )[1]
    original_max_link = float(np.max(original_links))
    for d in PLACEMENT_DISTANCES_A:
        candidate = saved_is.copy()
        candidate.positions[ads] += (d - length) * vector / length
        links = find_mic(
            candidate.positions[ads] - saved_fs.positions[ads],
            candidate.cell,
            candidate.pbc
        )[1]
        min_height = float(np.min(candidate.positions[ads, 2] - z0))
        collisions = _collisions(dump(candidate), ads, 0.7)
        reasons = (['BELOW_SURFACE'] if min_height < 0 else []) + (['EXISTING_COLLISION_PREDICATE'] if collisions else [])
        options.append(dict(
            distance_A=d,
            max_labeled_ads_link_A=float(np.max(links)),
            labeled_ads_links_A=[float(x) for x in links],
            min_ads_height_A=min_height,
            collisions=collisions,
            valid=not reasons,
            rejection_reasons=reasons
        ))
    valid = [x for x in options if x['valid']]
    return dict(
        chosen_distance_A=min(
            valid,
            key=lambda x: (x['max_labeled_ads_link_A'], x['distance_A'])
        )['distance_A'] if valid else None,
        options=options,
        status='READY' if valid else 'PLACEMENT_UNRESOLVED',
        original_saved_IS_max_labeled_ads_link_A=original_max_link,
        chosen_improves_max_link=min((x['max_labeled_ads_link_A'] for x in valid)) < original_max_link if valid else None,
        collision_scope='adsorbate_vs_nonadsorbate_and_nonzero_self_periodic_images_only',
        no_model_calls=True,
        original_saved_references_unchanged=True
    )
