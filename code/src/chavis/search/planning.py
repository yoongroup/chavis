import math


def numbers(values, name):
    values = list(values)
    if not values or any((type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError(f'{name}: nonempty finite numbers required')
    return values


def method1_seeds(
    energies,
    bonds,
    d_is,
    ref_bond,
    *,
    cap=12,
    window=0.5,
    ratios=(1.1, 1.8)
):
    energies, bonds = (numbers(energies, 'energies'), numbers(bonds, 'bonds'))
    if not isinstance(ratios, (tuple, list)) or len(ratios) != 2:
        raise ValueError('Bond ratio window requires two bounds')
    numbers([d_is, ref_bond, window, *ratios], 'selection parameters')
    if len(energies) != len(bonds) or not 3 <= len(energies) <= 40:
        raise ValueError('IDPP requires 3..40 matched images')
    if type(cap) is not int or not 1 <= cap <= 12 or min(bonds) <= 0 or (d_is <= 0) or (ref_bond <= 0) or (window < 0) or (not 0 < ratios[0] <= ratios[1]):
        raise ValueError('Invalid seed parameters')
    peak = max(range(1, len(energies) - 1), key=lambda i: (energies[i], -i))
    peak_window = {peak}
    for direction in (-1, 1):
        i = peak + direction
        while 0 < i < len(energies) - 1 and energies[peak] - energies[i] <= window:
            peak_window.add(i)
            i += direction
    bond_window = {i for i in range(1, len(bonds) - 1) if ratios[0] * d_is <= bonds[i] <= ratios[1] * d_is}
    indices = sorted(sorted(
        peak_window | bond_window,
        key=lambda i: (abs(bonds[i] - ref_bond), i)
    )[:cap])
    return dict(
        peak=peak,
        indices=indices,
        peak_window=sorted(peak_window),
        bond_window=sorted(bond_window),
        d_is_A=d_is,
        direction='neighbor_positions_difference',
        ci_neb_required=False
    )


def scan_targets(d_fs, d_is, step=0.1):
    numbers([d_fs, d_is, step], 'scan distances')
    if not 0 < d_is < d_fs or step != 0.1:
        raise ValueError('Method 2 requires decreasing positive bond and validated 0.1 A step')
    count = int(round((d_fs - d_is) / step))
    if count > 10000:
        raise ValueError('Scan exceeds bounded planner size')
    return [d_fs - k * step for k in range(count) if d_fs - k * step > d_is + 1e-12] + [d_is]


def method2_seeds(energies, bonds, d_is):
    energies, bonds = (numbers(energies, 'energies'), numbers(bonds, 'bonds'))
    numbers([d_is], 'd_is')
    if len(energies) != len(bonds) or d_is <= 0 or min(bonds) <= 0:
        raise ValueError('Invalid scan')
    peak = max(range(len(energies)), key=lambda i: (energies[i], -i))
    a = sorted(sorted(range(len(energies)), key=lambda i: (abs(i - peak), i))[:10])
    b = [
        i for i,
        bond in enumerate(bonds) if i not in a and 1.1 * d_is <= bond <= 2 * d_is
    ]
    return dict(peak=peak, stage_a=a, stage_b=b)


def stage_b_decision(expected_ids, results):
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Unique nonempty Stage A identities required')
    if any((results.get(i, {}).get('execution') != 'completed' for i in expected_ids)):
        return 'HOLD_INCOMPLETE_STAGE_A'
    scientific = {
        'TARGET_TS',
        'NOT_CONVERGED',
        'ZERO_IMAG',
        'MULTI_IMAG',
        'OTHER_CONNECTION',
        'CONNECTION_NOT_CONVERGED',
        'PHYSICS_REJECT',
        'ABORT_DIVERGE',
        'ABORT_SLAB',
        'ABORT_BELOW_SURFACE'
    }
    if any((results[i].get('verdict') not in scientific for i in expected_ids)):
        return 'HOLD_UNRESOLVED_STAGE_A'
    return 'SKIP_TARGET_FOUND' if any((results[i]['verdict'] == 'TARGET_TS' for i in expected_ids)) else 'RELEASE_STAGE_B'


def method1_plan():
    variants = [dict(
        id=f'{kind}_d{d:.1f}',
        distance_A=d,
        constrained_relax=kind == 'relax'
    ) for kind in ('rigid', 'relax') for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    bands = [dict(
        id=f"{v['id']}_N{n}",
        variant=v['id'],
        images=n
    ) for v in variants for n in (17, 30, 40)]
    return dict(
        variants=variants,
        bands=bands,
        maximum_seed_slots=len(bands) * 12,
        actual_seeds=None,
        ci_neb_required=False,
        note='432 is a capacity bound, not an observed seed count'
    )
