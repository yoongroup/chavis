import math
POLICY_VERSION = 'approach-0.1-v1'
ROUND_OFF_A = 1e-12


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def judge_progress(
    *,
    d_seed,
    d_now,
    d_ref,
    ref_range,
    rmsd,
    rmsd_limit=2.0,
    minimum_approach=0.1
):
    d_seed, d_now, d_ref, rmsd, rmsd_limit, minimum_approach = (
        _number(v, n) for v,
        n in zip(
            (d_seed, d_now, d_ref, rmsd, rmsd_limit, minimum_approach),
            ('d_seed', 'd_now', 'd_ref', 'rmsd', 'rmsd_limit', 'minimum_approach')
        )
    )
    if len(ref_range) != 2:
        raise ValueError('ref_range must have two bounds')
    lo, hi = (_number(x, 'ref_range') for x in ref_range)
    if min(d_seed, d_now, d_ref, rmsd, rmsd_limit, minimum_approach, lo) < 0 or lo > hi:
        raise ValueError('Invalid distances or range')
    improvement = abs(d_seed - d_ref) - abs(d_now - d_ref)
    inside = lo <= d_now <= hi
    approaching = improvement >= minimum_approach or math.isclose(
        improvement,
        minimum_approach,
        rel_tol=0,
        abs_tol=ROUND_OFF_A
    )
    abort = not inside and rmsd > rmsd_limit and (not approaching)
    reason = 'OUTSIDE_RANGE_NO_MINIMUM_APPROACH' if abort else 'BOND_IN_RANGE' if inside else 'MINIMUM_APPROACH_MET' if approaching else 'RMSD_WITHIN_LIMIT'
    return (
        'ABORT_DIVERGE' if abort else 'OK',
        {
            'policy_version': POLICY_VERSION,
            'reason': reason,
            'bond_seed_A': d_seed,
            'bond_A': d_now,
            'ref_bond_A': d_ref,
            'ref_bond_range_A': [lo, hi],
            'improvement_vs_original_seed_A': improvement,
            'minimum_approach_A': minimum_approach,
            'approaching': approaching,
            'bond_inside_ref_range': inside,
            'rmsd_ads_vs_seed_A': rmsd,
            'diverge_rmsd_A': rmsd_limit,
            'comparison_roundoff_A': ROUND_OFF_A
        }
    )


def judge_dimer_step(a, seed, IS, z0, rules, anchor, moving):
    import numpy as np
    from ase.geometry import find_mic
    from ase.constraints import FixAtoms
    ads = list(a.info['adsorbate_indices'])
    h = float(np.min(a.positions[ads, 2] - z0))
    measures = {'min_ads_height_A': h, 'policy_version': POLICY_VERSION}
    if h < rules['h_min_A']:
        return (
            'ABORT_BELOW_SURFACE',
            {**measures, 'reason': 'ADSORBATE_BELOW_REFERENCE_PLANE'}
        )
    fixed = {int(i) for c in IS.constraints if isinstance(c, FixAtoms) for i in c.get_indices()}
    profile = IS.info.get('surface_profile')
    if isinstance(profile, dict) and profile.get('element') == 'Cu':
        from chavis.judgment.surface_context import selected_surface_element
        from .geometry import from_ase
        if selected_surface_element(from_ase(IS)) != 'Cu':
            raise ValueError('Unsupported Cu slab context in Dimer judge')
        metal = 'Cu'
    else:
        metal = 'Ni'
    free_metal = [
        i for i,
        atom in enumerate(IS) if atom.symbol == metal and i not in fixed
    ]
    disp = find_mic(
        a.positions[free_metal] - IS.positions[free_metal],
        IS.cell,
        IS.pbc
    )[1]
    slab = float(np.max(disp)) if free_metal else 0.0
    measures['max_free_Ni_disp_vs_IS_A' if metal == 'Ni' else 'max_free_Cu_disp_vs_IS_A'] = slab
    if slab > rules['slab_max_A']:
        return (
            'ABORT_SLAB',
            {
                **measures,
                'reason': 'FREE_NI_DISPLACEMENT' if metal == 'Ni' else 'FREE_CU_DISPLACEMENT'
            }
        )
    distances = find_mic(a.positions[ads] - seed.positions[ads], a.cell, a.pbc)[1]
    rmsd = float(np.sqrt(np.mean(np.asarray(distances) ** 2)))
    if rules.get('policy') == 'saved-is-window-v1':
        from .return_policy import interval_progress
        d_seed = float(seed.get_distance(anchor, moving, mic=True))
        d_now = float(a.get_distance(anchor, moving, mic=True))
        progress = interval_progress(
            d_seed,
            d_now,
            float(IS.get_distance(anchor, moving, mic=True))
        )
        inside = progress['distance_now_A'] == 0
        abort = not inside and rmsd > rules['diverge_rmsd_A'] and (not progress['approached'])
        return (
            'ABORT_DIVERGE' if abort else 'OK',
            {
                **measures,
                'policy_version': 'saved-is-window-v1',
                'reason': 'OUTSIDE_WINDOW_NO_MINIMUM_APPROACH' if abort else 'BOND_IN_WINDOW' if inside else 'MINIMUM_APPROACH_MET' if progress['approached'] else 'RMSD_WITHIN_LIMIT',
                'bond_seed_A': d_seed,
                'bond_A': d_now,
                'original_saved_IS_r0_A': float(IS.get_distance(anchor, moving, mic=True)),
                'interval_progress': progress,
                'rmsd_ads_vs_seed_A': rmsd,
                'diverge_rmsd_A': rules['diverge_rmsd_A']
            }
        )
    code, progress = judge_progress(
        d_seed=float(seed.get_distance(anchor, moving, mic=True)),
        d_now=float(a.get_distance(anchor, moving, mic=True)),
        d_ref=rules['ref_bond_A'],
        ref_range=rules['ref_bond_range_A'],
        rmsd=rmsd,
        rmsd_limit=rules['diverge_rmsd_A'],
        minimum_approach=rules.get('minimum_approach_A', 0.1)
    )
    return (code, {**measures, **progress})
