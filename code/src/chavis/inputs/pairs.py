from __future__ import annotations
import numpy as np
SAME_ADS_A = 0.1
SAME_SLAB_A = 0.1
IS_FAR_ABOVE_EV = 1.0
ROUTES = {'T2': 'scan_first', 'T0-H': 'direct_then_scan', 'T0-F': 'direct_then_scan'}


def _groups(symbols):
    g = {}
    for i, s in enumerate(symbols):
        g.setdefault(s, []).append(i)
    return list(g.values())


def _mic(a, b, cell):
    from ase.geometry import find_mic
    d = a[:, None, :] - b[None, :, :]
    return find_mic(
        d.reshape(-1, 3),
        cell,
        pbc=[True, True, False]
    )[1].reshape(len(a), len(b))


def plain_rmsd(a, b, cell):
    from ase.geometry import find_mic
    d = find_mic(
        np.asarray(b, float) - np.asarray(a, float),
        cell,
        pbc=[True, True, False]
    )[1]
    return float(np.sqrt(np.mean(d ** 2)))


def slab_permutations(clean, symbols, ops, cell):
    from scipy.optimize import linear_sum_assignment
    out = []
    for r, t in ops:
        moved = clean @ r.T + t
        perm = np.empty(len(clean), int)
        for idx in _groups(symbols):
            c = _mic(moved[idx], clean[idx], cell)
            ri, ci = linear_sum_assignment(c)
            if c[ri, ci].max() > 0.05:
                raise ValueError('operation does not map the clean slab onto itself')
            perm[np.asarray(idx)[ri]] = np.asarray(idx)[ci]
        out.append(perm)
    return out


def best_mapping(
    ads_ref,
    ads_mov,
    symbols,
    ops,
    cell,
    slab_ref=None,
    slab_mov=None,
    sperm=None
):
    from scipy.optimize import linear_sum_assignment
    ads_ref, ads_mov = (np.asarray(ads_ref, float), np.asarray(ads_mov, float))
    best = (np.inf, None, None)
    for oi, (r, t) in enumerate(ops):
        if slab_ref is not None:
            from ase.geometry import find_mic
            ds = find_mic(
                np.asarray(slab_mov) @ r.T + t - np.asarray(slab_ref)[sperm[oi]],
                cell,
                pbc=[True, True, False]
            )[1]
            if ds.max() >= SAME_SLAB_A:
                continue
        m = ads_mov @ r.T + t
        perm = np.empty(len(symbols), int)
        total = 0.0
        for idx in _groups(symbols):
            c = _mic(m[idx], ads_ref[idx], cell)
            ri, ci = linear_sum_assignment(c)
            total += float(np.sum(c[ri, ci] ** 2))
            perm[np.asarray(idx)[ri]] = np.asarray(idx)[ci]
        rmsd = (total / len(symbols)) ** 0.5
        if rmsd < best[0]:
            best = (rmsd, oi, perm)
    return best


def to_frame(slab_pos, ads_pos, op, slab_perm, ads_perm, cell):
    r, t = op
    s = np.asarray(slab_pos, float) @ r.T + t
    a = np.asarray(ads_pos, float) @ r.T + t
    s2 = np.empty_like(s)
    s2[slab_perm] = s
    a2 = np.empty_like(a)
    a2[ads_perm] = a
    inv = np.linalg.inv(cell)
    for x in (s2, a2):
        f = x @ inv
        f[:, :2] -= np.floor(f[:, :2])
        x[:] = f @ cell
    return (s2, a2)


def pair_reaction(sp, reaction, is_rows, fs_rows, log=print, exclude_intercalated=True):
    from chavis.inputs.is_refine import intercalated
    from chavis.surface.symmetry import surface_operations
    ops = surface_operations(sp.slab)
    cell = sp.slab.cell.array
    slab_sym = sp.slab.get_chemical_symbols()
    sperm = slab_permutations(sp.clean, slab_sym, ops, cell)
    pairs, derived, new_is = ([], [], [])
    for fi, f in enumerate(fs_rows):
        if not (f['product_match'] and f['converged'] and (f['energy_eV'] is not None)):
            continue
        if exclude_intercalated and intercalated(f):
            continue
        cand = []
        for i in is_rows:
            rmsd, oi, perm = best_mapping(
                i['final_adsorbate_positions'],
                f['final_adsorbate_positions'],
                sp.sym,
                ops,
                cell
            )
            cand.append((rmsd, i, oi, perm))
        own = [c for c in cand if c[1]['minimum_representative'] == f.get('from_is')]
        if own:
            i = own[0][1]
            rmsd = plain_rmsd(
                i['final_adsorbate_positions'],
                f['final_adsorbate_positions'],
                cell
            )
            s2, a2 = (
                np.asarray(f['final_slab_positions'], float),
                np.asarray(f['final_adsorbate_positions'], float)
            )
            oi, perm, how = (None, np.arange(len(sp.sym)), 'derived_from_is')
        else:
            from chavis.inputs.is_from_fs import derive_is
            fix = f.get(
                'fixed_layers',
                is_rows[0]['metal']['fixed_layers'] if is_rows else None
            )
            if fix is None:
                raise ValueError('spontaneous FS must record fixed_layers when no IS is available')
            d = derive_is(sp, f, fix, log)
            derived.append({
                'fs_index': fi,
                'fs_source': f['source'],
                **{
                    k: v for k,
                    v in d.items() if k not in (
                        'final_slab_positions',
                        'final_adsorbate_positions'
                    )
                }
            })
            if d['outcome'] in ('dissociated', 'error') or not d['converged']:
                log(f"  {reaction['id']} FS {fi}: derived IS did not come back as an intact molecule ({d['outcome']}), no pair")
                derived[-1]['result'] = 'derived_from_fs_dissociated_again'
                continue
            if exclude_intercalated and intercalated(d):
                log(f"  {reaction['id']} FS {fi}: derived IS is intercalated, no pair")
                derived[-1]['result'] = 'derived_from_fs_intercalated'
                derived[-1]['reason'] = 'intercalated_derived_is'
                continue
            same = []
            for candidate in is_rows + new_is:
                if candidate['outcome'] != d['outcome'] or abs(candidate['energy_eV'] - d['energy_eV']) >= 0.05:
                    continue
                mapping = best_mapping(
                    candidate['final_adsorbate_positions'],
                    d['final_adsorbate_positions'],
                    sp.sym,
                    ops,
                    cell,
                    candidate['final_slab_positions'],
                    d['final_slab_positions'],
                    sperm
                )
                if mapping[0] < SAME_ADS_A:
                    same.append((candidate, mapping))
            if same:
                i, (_, oi, perm) = same[0]
                s2, a2 = to_frame(
                    f['final_slab_positions'],
                    f['final_adsorbate_positions'],
                    ops[oi],
                    sperm[oi],
                    perm,
                    cell
                )
                rmsd = plain_rmsd(i['final_adsorbate_positions'], a2, cell)
                how = 'derived_is_matches_existing'
            else:
                i = {
                    'minimum_representative': f'derived_from_fs_{fi}',
                    'energy_eV': d['energy_eV'],
                    'metal': {'fixed_layers': fix},
                    'outcome': d['outcome'],
                    'labels': d.get('labels') or {},
                    'final_slab_positions': d['final_slab_positions'],
                    'final_adsorbate_positions': d['final_adsorbate_positions']
                }
                s2, a2 = (
                    np.asarray(f['final_slab_positions'], float),
                    np.asarray(f['final_adsorbate_positions'], float)
                )
                oi, perm, how = (None, np.arange(len(sp.sym)), 'derived_from_fs_new_is')
                rmsd = plain_rmsd(i['final_adsorbate_positions'], a2, cell)
                new_is.append(i)
            derived[-1]['result'] = how
            derived[-1]['paired_is'] = i['minimum_representative']
        pairs.append({
            'reaction': reaction['id'],
            'route': ROUTES.get(reaction.get('default_route'), 'unknown'),
            'default_route': reaction.get('default_route'),
            'paired_is': i['minimum_representative'],
            'is_energy_eV': i['energy_eV'],
            'fs_index': fi,
            'fs_source': f['source'],
            'from_is': f.get('from_is'),
            'fs_energy_eV': f['energy_eV'],
            'reaction_energy_eV': round(f['energy_eV'] - i['energy_eV'], 4),
            'pairing': how,
            'pairing_rmsd_A': round(rmsd, 4),
            'pairing_op': oi,
            'ads_permutation': perm.tolist(),
            'rmsd_to_other_is_A': sorted((round(c[0], 4) for c in cand if c[1] is not i)),
            'fixed_layers': i['metal']['fixed_layers'],
            'is_slab_positions': i['final_slab_positions'],
            'is_adsorbate_positions': i['final_adsorbate_positions'],
            'fs_slab_positions': s2.round(5).tolist(),
            'fs_adsorbate_positions': a2.round(5).tolist()
        })
    e_low = min(
        (x['energy_eV'] for x in list(is_rows) + new_is if x.get('energy_eV') is not None),
        default=None
    )
    for p in pairs:
        d = None if e_low is None else p['is_energy_eV'] - e_low
        p['is_above_lowest_is_eV'] = None if d is None else round(d, 4)
        p['is_far_above_lowest'] = bool(d is not None and d > IS_FAR_ABOVE_EV)
    dedupe_pairs(pairs, sp, ops, sperm, cell)
    return (pairs, derived)


def pair_equivalent(
    p,
    q,
    symbols,
    ops,
    sperm,
    cell,
    tol_ads=SAME_ADS_A,
    tol_slab=SAME_SLAB_A
):
    from scipy.optimize import linear_sum_assignment
    pi, pf = (
        np.asarray(p['is_adsorbate_positions']),
        np.asarray(p['fs_adsorbate_positions'])
    )
    qi, qf = (
        np.asarray(q['is_adsorbate_positions']),
        np.asarray(q['fs_adsorbate_positions'])
    )
    psi, psf = (np.asarray(p['is_slab_positions']), np.asarray(p['fs_slab_positions']))
    qsi, qsf = (np.asarray(q['is_slab_positions']), np.asarray(q['fs_slab_positions']))
    n = len(symbols)
    for oi, (r, t) in enumerate(ops):
        mi, mf = (pi @ r.T + t, pf @ r.T + t)
        ti = tf = 0.0
        for idx in _groups(symbols):
            ci, cf = (_mic(mi[idx], qi[idx], cell), _mic(mf[idx], qf[idx], cell))
            ri, cj = linear_sum_assignment(ci ** 2 + cf ** 2)
            ti += float(np.sum(ci[ri, cj] ** 2))
            tf += float(np.sum(cf[ri, cj] ** 2))
        if (ti / n) ** 0.5 >= tol_ads or (tf / n) ** 0.5 >= tol_ads:
            continue
        perm = sperm[oi]
        ok = True
        for a, b in ((psi, qsi), (psf, qsf)):
            m = a @ r.T + t
            from ase.geometry import find_mic
            if find_mic(
                m - b[perm],
                cell,
                pbc=[True, True, False]
            )[1].max() >= tol_slab:
                ok = False
                break
        if ok:
            return True
    return False


def dedupe_pairs(pairs, sp, ops, sperm, cell):
    reps = []
    for p in sorted(
        pairs,
        key=lambda x: (x['fs_energy_eV'] - x['is_energy_eV'], x['pairing_rmsd_A'])
    ):
        for g, rep in enumerate(reps):
            if abs(rep['is_energy_eV'] - p['is_energy_eV']) < 0.05 and abs(rep['fs_energy_eV'] - p['fs_energy_eV']) < 0.05 and pair_equivalent(rep, p, sp.sym, ops, sperm, cell):
                p['pair_group'] = g
                p['pair_representative'] = False
                rep['pair_group_sources'] = sorted(set(rep['pair_group_sources']) | {p['fs_source']})
                break
        else:
            p['pair_group'] = len(reps)
            p['pair_representative'] = True
            p['pair_group_sources'] = [p['fs_source']]
            reps.append(p)
    return pairs
