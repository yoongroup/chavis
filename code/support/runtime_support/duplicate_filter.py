from __future__ import annotations
import copy, math, time
from collections import Counter, defaultdict
from geometry import validate, parse_indices, norm, sub, basis, dot
from workbench_tools import displacement
I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _ads(s):
    val = s.get('adsorbate_indices', s.get('meta', {}).get('adsorbate_indices'))
    if val is None:
        raise ValueError('Explicit adsorbate_indices required for duplicate filtering. Set them in the viewer first.')
    return parse_indices(val, len(s['symbols']))


def _labels(s):
    if s.get('meta', {}).get('unsupported_constraints'):
        raise ValueError('Duplicate filtering cannot validate these constraints. Review unsupported_constraints before pruning.')
    ads = set(_ads(s))
    fixed = set(s.get('fixed_indices', []))
    mom = s.get('initial_magmoms')
    if mom is not None and (len(mom) != len(s['symbols']) or any((isinstance(x, (list, tuple)) for x in mom))):
        raise ValueError('Noncollinear/malformed magnetic moments are not supported by this geometry filter.')
    labels = []
    for i, sym in enumerate(s['symbols']):
        labels.append((
            sym,
            i in ads,
            i in fixed,
            round(float(mom[i]), 6) if mom is not None else None
        ))
    return labels


def _apply(positions, op):
    m = op['cart_matrix']
    t = op['cart_translation']
    return [[sum((p[k] * m[k][j] for k in range(3))) + t[j] for j in range(3)] for p in positions]


def _identity():
    return {
        'cart_matrix': copy.deepcopy(I3),
        'cart_translation': [0.0, 0.0, 0.0],
        'fractional_rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'fractional_translation': [0.0, 0.0, 0.0],
        'determinant': 1,
        'kind': 'identity',
        'source': 'identity'
    }


def _raw_distances(a, b, s):
    return [norm(displacement(pa, pb, s)) for pa, pb in zip(a, b)]


def _cost(a, b, cell, pbc):
    import numpy as np
    try:
        from ase.geometry import find_mic
    except ImportError:
        geom = {'cell': cell, 'pbc': pbc}
        return np.asarray([[norm(displacement(pa, pb, geom)) for pb in b] for pa in a])
    dif = np.asarray(a)[:, None, :] - np.asarray(b)[None, :, :]
    shape = dif.shape[:2]
    _, dist = find_mic(dif.reshape(-1, 3), cell=cell, pbc=[pbc[0], pbc[1], False])
    return np.asarray(dist).reshape(shape)


def _assignment(aa, bb, la, lb, s, permutation, ids_a, ids_b):
    n = len(la)
    if Counter(la) != Counter(lb):
        return None
    if not permutation:
        lookup = {i: j for j, i in enumerate(ids_b)}
        mapping = [lookup.get(i, -1) for i in ids_a]
        if any((j < 0 or la[i] != lb[j] for i, j in enumerate(mapping))):
            return None
        return mapping
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ValueError('Element-permutation matching requires NumPy/SciPy. Run INSTALL_SCIENCE.command.') from e
    groups_a = defaultdict(list)
    groups_b = defaultdict(list)
    for i, key in enumerate(la):
        groups_a[key].append(i)
    for i, key in enumerate(lb):
        groups_b[key].append(i)
    mapping = [-1] * n
    for key, ia in groups_a.items():
        ib = groups_b[key]
        cost = np.zeros((len(ia), len(ib)))
        for fa, fb in zip(aa, bb):
            d = _cost([fa[i] for i in ia], [fb[j] for j in ib], s['cell'], s['pbc'])
            cost += d * d
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            mapping[ia[int(i)]] = ib[int(j)]
    return mapping


def _surface_ops(s, options):
    if options['mode'] == 'rmsd':
        return [_identity()]
    for key in ('soc', 'external_field'):
        value = s.get(key, s.get('meta', {}).get(key))
        nonzero = any((bool(v) for v in value)) if isinstance(value, (list, tuple)) else bool(value)
        if nonzero:
            raise ValueError('Surface symmetry review does not support SOC or symmetry-breaking fields. Use direct RMSD or a custom symmetry model.')
    try:
        import numpy as np
        import spglib
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ValueError('Surface symmetry/mirror review requires NumPy, SciPy and spglib. Run INSTALL_SCIENCE.command. No fake symmetry fallback.') from e
    if not all(s['pbc'][:2]):
        raise ValueError('Surface symmetry mode requires both in-plane periodic directions.')
    cell = np.asarray(s['cell'], float)
    if abs(np.linalg.det(cell)) < 1e-08:
        raise ValueError('Surface symmetry needs a full cell including vacuum.')
    inv = np.linalg.inv(cell)
    labels = _labels(s)
    slab = [i for i, x in enumerate(labels) if not x[1]]
    if not slab:
        raise ValueError('No substrate atoms: surface symmetry mode cannot be used for a molecule alone.')
    slabels = [labels[i] for i in slab]
    unique = {k: i + 1 for i, k in enumerate(sorted(set(slabels), key=repr))}
    pos = np.asarray([s['positions'][i] for i in slab])
    frac = pos @ inv
    sym = spglib.get_symmetry(
        (cell, frac, [unique[k] for k in slabels]),
        symprec=options['symprec']
    )
    if sym is None:
        raise ValueError('spglib could not find slab symmetry. Use direct RMSD or review substrate selection.')
    normal = np.asarray(basis(s)[2])
    ops = []
    for rot, t in zip(sym['rotations'], sym['translations']):
        m = inv @ np.asarray(rot).T @ cell
        trans = np.asarray(t) @ cell
        det = round(float(np.linalg.det(m)))
        if det < 0 and (not options['mirror']):
            continue
        if not np.allclose(m @ m.T, np.eye(3), atol=1e-05):
            continue
        if not np.allclose(normal @ m, normal, atol=1e-05):
            continue
        op = {
            'cart_matrix': m.tolist(),
            'cart_translation': trans.tolist(),
            'fractional_rotation': np.asarray(rot).tolist(),
            'fractional_translation': np.asarray(t).tolist(),
            'determinant': det,
            'kind': 'mirror' if det < 0 else 'surface_rotation_or_translation',
            'source': 'spglib.get_symmetry'
        }
        transformed = _apply(pos.tolist(), op)
        mp = _assignment(
            [transformed],
            [pos.tolist()],
            slabels,
            slabels,
            s,
            True,
            list(range(len(slab))),
            list(range(len(slab)))
        )
        if mp is None:
            continue
        distances = _raw_distances(transformed, [pos[j].tolist() for j in mp], s)
        if max(distances, default=0) > options['symprec']:
            continue
        ops.append(op)
    if not ops:
        return [_identity()]
    ops.sort(key=lambda op: (
        op['kind'] == 'mirror',
        sum((abs(op['cart_matrix'][i][j] - I3[i][j]) for i in range(3) for j in range(3))) + norm(op['cart_translation'])
    ))
    return ops


''


def _frames(item, pair):
    s = validate(item['structure'])
    seq = [s]
    if pair:
        parent = item.get('input_parent')
        if parent is None:
            raise ValueError('Paired-endpoint filtering requires input_parent on every selected candidate.')
        p = validate(parent)
        if p['atom_ids'] != s['atom_ids'] or p['symbols'] != s['symbols']:
            raise ValueError('IS/FS atom IDs or ordering differ within a candidate.')
        seq = [p, s]
        if item.get('path'):
            seq += [validate(f) for f in item['path']]
    return seq


def _edit_set(item, mapping=None):
    ed = item.get('detail', {}).get('reaction_edits') or item.get('sourceParams', {}).get('reaction_edits')
    if not ed:
        return None
    out = {}
    for key in ('break', 'form'):
        out[key] = sorted((tuple(sorted((mapping[int(i)] if mapping is not None else int(i) for i in pair))) for pair in ed.get(key, [])))
    return out


def compare_items(a, b, ops, options):
    pair = options['pair']
    aa = _frames(a, pair)
    bb = _frames(b, pair)
    if len(aa) != len(bb):
        return None
    sa = aa[0]
    sb = bb[0]
    if len(sa['symbols']) != len(sb['symbols']) or sa['pbc'] != sb['pbc']:
        return None
    if any((
        abs(x - y) > options['cell_tol'] for rowa,
        rowb in zip(sa['cell'], sb['cell']) for x,
        y in zip(rowa, rowb)
    )):
        return None
    la = _labels(sa)
    lb = _labels(sb)
    if Counter(la) != Counter(lb):
        return None
    for ka, kb in zip(aa, bb):
        for key in (
            'charge',
            'calculator_profile',
            'model_id',
            'xc',
            'soc',
            'external_field'
        ):
            va = ka.get(key, ka.get('meta', {}).get(key))
            vb = kb.get(key, kb.get('meta', {}).get(key))
            if va != vb:
                return None
    ads = [i for i, k in enumerate(la) if k[1]]
    slab = [i for i, k in enumerate(la) if not k[1]]
    if not ads:
        raise ValueError('No adsorbate atoms to calculate adsorbate RMSD.')
    for f in aa:
        if _labels(f) != la or f['pbc'] != sa['pbc'] or f['cell'] != sa['cell']:
            return None
    for f in bb:
        if _labels(f) != lb or f['pbc'] != sb['pbc'] or f['cell'] != sb['cell']:
            return None
    tol = options['rmsd']
    max_tol = options['max_atom']
    for op in ops:
        ap = [_apply(f['positions'], op) for f in aa]
        bp = [f['positions'] for f in bb]
        mp = _assignment(
            ap,
            bp,
            la,
            lb,
            sa,
            options['permutation'],
            sa['atom_ids'],
            sb['atom_ids']
        )
        if mp is None:
            continue
        ea = _edit_set(a, mp)
        eb = _edit_set(b)
        if pair and (ea is not None or eb is not None) and (ea != eb):
            continue
        frame_rmsd = []
        max_a = 0.0
        max_s = 0.0
        valid = True
        for fa, fb in zip(ap, bp):
            ds = _raw_distances(fa, [fb[j] for j in mp], sa)
            r = math.sqrt(sum((ds[i] ** 2 for i in ads)) / len(ads))
            mx = max((ds[i] for i in ads))
            sm = max((ds[i] for i in slab), default=0.0)
            if r > tol or mx > max_tol or sm > options['symprec']:
                valid = False
                break
            frame_rmsd.append(r)
            max_a = max(max_a, mx)
            max_s = max(max_s, sm)
        if not valid:
            continue
        if pair:
            da = [sub(ap[1][i], ap[0][i]) for i in range(len(mp))]
            db = [sub(bp[1][mp[i]], bp[0][mp[i]]) for i in range(len(mp))]
            if max((norm(sub(x, y)) for x, y in zip(da, db))) > 2 * max_tol:
                continue
        if options.get('energy_guard') is not None:
            cutoff = float(options['energy_guard'])
            if any((
                'energy' in fa and 'energy' in fb and (abs(fa['energy'] - fb['energy']) > cutoff) for fa,
                fb in zip(aa, bb)
            )):
                continue
        return {
            'rmsd_A': max(frame_rmsd),
            'frame_rmsd_A': frame_rmsd,
            'max_adsorbate_displacement_A': max_a,
            'max_slab_displacement_A': max_s,
            'mapping': mp,
            'operation': op,
            'paired': pair,
            'kind': op['kind'],
            'matched_frames': len(ap)
        }
    return None


def review_duplicates(items, opts=None):
    opts = opts or {}
    options = {
        'rmsd': 0.05,
        'max_atom': 0.1,
        'symprec': 0.02,
        'cell_tol': 1e-06,
        'mode': 'rmsd',
        'mirror': False,
        'permutation': False,
        'pair': True,
        'energy_guard': 0.02,
        **opts
    }
    for key in ('rmsd', 'max_atom', 'symprec', 'cell_tol'):
        options[key] = float(options[key])
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(key + ' must be positive and finite.')
    if options['mode'] not in ('rmsd', 'surface'):
        raise ValueError('Choose direct RMSD or surface symmetry mode.')
    if options['mirror'] and options['mode'] != 'surface':
        raise ValueError('Mirrors require verified surface-symmetry mode.')
    if len(items) > 300:
        raise ValueError('Review at most 300 structures per batch.')
    if not items:
        return {'groups': [], 'duplicates': [], 'options': options, 'count': 0}
    ordered = sorted(
        enumerate(items),
        key=lambda z: ({'preferred': 0, 'keep': 1}.get(z[1].get('review'), 2), z[0])
    )
    reps = []
    groups = []
    dups = []
    skipped = []
    started = time.monotonic()
    op_cache = {}
    for idx, item in ordered:
        if time.monotonic() - started > 80:
            raise ValueError('Review exceeded 80 s; reduce the selected set. No changes were applied.')
        if item.get('review') == 'reject' or item.get('accepted') is False:
            skipped.append({
                'id': item['id'],
                'reason': 'manually/geometrically rejected; not used as representative'
            })
            continue
        seq = _frames(item, options['pair'])
        base = seq[0]
        _labels(base)
        match = None
        for ri, gidx in reps:
            rep = items[ri]
            ref = _frames(rep, options['pair'])[0]
            if Counter(ref['symbols']) != Counter(base['symbols']):
                continue
            if ri not in op_cache:
                op_cache[ri] = _surface_ops(ref, options)
            witness = compare_items(rep, item, op_cache[ri], options)
            if witness:
                match = {
                    'id': item['id'],
                    'name': item.get('name', ''),
                    'representative_id': rep['id'],
                    'representative_name': rep.get('name', ''),
                    **witness
                }
                break
        if match:
            dups.append(match)
            groups[gidx]['members'].append(item['id'])
        else:
            reps.append((idx, len(groups)))
            groups.append({
                'representative_id': item['id'],
                'name': item.get('name', ''),
                'members': [item['id']]
            })
    return {
        'options': options,
        'count': len(items),
        'unique_count': len(reps),
        'groups': groups,
        'duplicates': dups,
        'skipped': skipped,
        'elapsed_s': time.monotonic() - started,
        'warning': 'Geometric duplicate REVIEW only. No files/atoms deleted; no energetic/path equivalence guarantee. No free adsorbate Kabsch alignment. Symmetry assumes collinear scalar magnetic labels, no SOC/external symmetry-breaking fields.'
    }
