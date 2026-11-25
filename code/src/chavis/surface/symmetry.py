from __future__ import annotations
import numpy as np
SYMPREC_A = 0.01


def surface_operations(atoms, symprec=SYMPREC_A):
    import spglib
    cell = atoms.cell.array
    ds = spglib.get_symmetry(
        (cell, atoms.get_scaled_positions(), atoms.numbers),
        symprec=symprec
    )
    ops = []
    inv = np.linalg.inv(cell)
    for rot, tr in zip(ds['rotations'], ds['translations']):
        if list(rot[2]) != [0, 0, 1] or abs(tr[2] - round(tr[2])) > 1e-06:
            continue
        r_cart = cell.T @ rot @ inv.T
        t_cart = cell.T @ tr
        ops.append((r_cart, t_cart))
    if not ops:
        raise ValueError('no surface symmetry operation found (identity missing)')
    return ops


def _mic_cost(a, b, cell):
    from ase.geometry import find_mic
    d = a[:, None, :] - b[None, :, :]
    _, dist = find_mic(d.reshape(-1, 3), cell, pbc=[True, True, False])
    return dist.reshape(len(a), len(b))


def equivalent(pos_a, pos_b, symbols, ops, cell, tol=0.1):
    from scipy.optimize import linear_sum_assignment
    pos_a, pos_b = (np.asarray(pos_a, float), np.asarray(pos_b, float))
    groups = {}
    for i, s in enumerate(symbols):
        groups.setdefault(s, []).append(i)
    n = len(symbols)
    for r, t in ops:
        a2 = pos_a @ r.T + t
        total = 0.0
        for idx in groups.values():
            c = _mic_cost(a2[idx], pos_b[idx], cell)
            ri, ci = linear_sum_assignment(c)
            total += float(np.sum(c[ri, ci] ** 2))
            if total > tol * tol * n:
                break
        if (total / n) ** 0.5 < tol:
            return True
    return False


def max_rotation_order(ops):
    best = 1
    for r, _ in ops:
        xy = r[:2, :2]
        if np.linalg.det(xy) < 0:
            continue
        theta = abs(np.degrees(np.arctan2(xy[1, 0], xy[0, 0])))
        if theta < 0.001:
            continue
        n = 360.0 / theta
        if abs(n - round(n)) < 0.001:
            best = max(best, int(round(n)))
    return best


def azimuth_step(ops, max_step_deg=45.0):
    import math
    n = max_rotation_order(ops)
    k = math.ceil(360.0 / (n * max_step_deg))
    return 360.0 / (n * k)


def resolve_azimuth_step(requested, ops):
    """Keep the legacy auto grid, or validate an explicit slab-closed grid."""
    from chavis.runtime.adsorption_settings import parse_azimuth_step
    step = parse_azimuth_step(requested)
    if step == 'auto':
        return azimuth_step(ops)
    for rotation, _ in ops:
        xy = rotation[:2, :2]
        if np.linalg.det(xy) < 0:
            continue
        angle = float(np.degrees(np.arctan2(xy[1, 0], xy[0, 0])))
        if abs(angle - round(angle / step) * step) > 0.001:
            raise ValueError(
                f'adsorption_azimuth_step_deg={step:g} is not closed under '
                f'the slab rotation {angle:g} degrees; use auto or a compatible divisor'
            )
    return step


def equivalent_with_slab(
    ads_a,
    ads_b,
    ads_symbols,
    slab_a,
    slab_b,
    slab_symbols,
    ops,
    cell,
    tol_ads=0.1,
    tol_slab=0.1
):
    from scipy.optimize import linear_sum_assignment
    ads_a, ads_b = (np.asarray(ads_a, float), np.asarray(ads_b, float))
    slab_a, slab_b = (np.asarray(slab_a, float), np.asarray(slab_b, float))
    ga, gs = ({}, {})
    for i, s in enumerate(ads_symbols):
        ga.setdefault(s, []).append(i)
    for i, s in enumerate(slab_symbols):
        gs.setdefault(s, []).append(i)
    n = len(ads_symbols)
    for r, t in ops:
        a2 = ads_a @ r.T + t
        total = 0.0
        for idx in ga.values():
            c = _mic_cost(a2[idx], ads_b[idx], cell)
            ri, ci = linear_sum_assignment(c)
            total += float(np.sum(c[ri, ci] ** 2))
        if (total / n) ** 0.5 >= tol_ads:
            continue
        s2 = slab_a @ r.T + t
        worst = 0.0
        for idx in gs.values():
            c = _mic_cost(s2[idx], slab_b[idx], cell)
            ri, ci = linear_sum_assignment(c)
            worst = max(worst, float(c[ri, ci].max()))
        if worst < tol_slab:
            return True
    return False
