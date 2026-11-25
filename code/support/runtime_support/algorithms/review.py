from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from collections import Counter
from geometry import validate, compatible, basis, dot, cross, norm, sub
from duplicate_filter import compare_items, _surface_ops, _labels, _apply
from workbench_tools import displacement


def snapshot(items):
    fields = (
        'id',
        'structure',
        'input_parent',
        'path',
        'path_direction',
        'parent_side',
        'generation_edits',
        'reaction_edits',
        'detail',
        'review'
    )
    payload = [{k: item.get(k) for k in fields} for item in items]
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False
    ).encode()).hexdigest()


def normalise_item(item):
    c = copy.deepcopy(item)
    if not c.get('input_parent'):
        raise ValueError('Reaction-pair duplicate checking requires each candidate parent structure.')
    c['structure'], c['input_parent'] = (
        validate(c['structure']),
        validate(c['input_parent'])
    )
    compatible(c['input_parent'], c['structure'])
    for frame in c.get('path') or []:
        compatible(c['input_parent'], validate(frame))
    if c.get('path_direction') == 'reactant_to_product' and c.get('parent_side') == 'product':
        c['path'] = list(reversed(c['path']))
    elif c.get('path') and c.get('path_direction', 'parent_to_opposite') not in (
        'parent_to_opposite',
        'reactant_to_product'
    ):
        raise ValueError('Cannot determine stored pathway direction.')
    c.setdefault('detail', {})['reaction_edits'] = copy.deepcopy(c.get('generation_edits') or c['detail'].get('generation_edits') or c['detail'].get('reaction_edits'))
    return c


def explain(witness, reference):
    op = witness['operation']
    m = op['cart_matrix']
    x, _, normal = basis(reference)
    moved_x = [sum((x[k] * m[k][j] for k in range(3))) for j in range(3)]
    angle = math.degrees(math.atan2(dot(cross(x, moved_x), normal), dot(x, moved_x)))
    shift = norm(op['cart_translation'])
    parts = []
    if op.get('determinant', 1) < 0:
        parts.append('mirror symmetry preserving surface orientation')
    elif abs(angle) > 0.001:
        parts.append(f'surface rotation {angle:.2f} degrees')
    if shift > 0.001:
        parts.append(f'{shift:.3f} Angstrom translation')
    if not parts:
        parts.append('compare in the same surface coordinate system')
    if any((i != j for i, j in enumerate(witness['mapping']))):
        parts.append('compare using element-preserving atom correspondence')
    parts.append('use the same correspondence and transform for parent and separated structures')
    return ' · '.join(parts)


def review(items, options=None):
    opts = {
        'mode': 'rmsd',
        'permutation': True,
        'pair': True,
        'mirror': False,
        'rmsd': 0.05,
        'max_atom': 0.1,
        'symprec': 0.02,
        'cell_tol': 1e-06,
        'energy_guard': 0.02,
        **(options or {})
    }
    if opts['pair'] is not True:
        raise ValueError('v1.0 checks parent and separated structures together; FS alone does not determine removal.')
    for key in ('permutation', 'pair', 'mirror'):
        if not isinstance(opts[key], bool):
            raise ValueError(f'{key} setting must be true/false.')
    if opts['mode'] not in ('rmsd', 'surface') or (opts['mirror'] and opts['mode'] != 'surface'):
        raise ValueError('Mirror reflection is restricted to surface symmetry checking.')
    for key in ('rmsd', 'max_atom', 'symprec', 'cell_tol'):
        opts[key] = float(opts[key])
        if not math.isfinite(opts[key]) or opts[key] <= 0:
            raise ValueError('Tolerance must be positive and finite.')
    if opts['energy_guard'] is not None and (not math.isfinite(float(opts['energy_guard'])) or float(opts['energy_guard']) < 0):
        raise ValueError('Energy tolerance must be at least 0.')
    if len(items) > 5000:
        raise ValueError('At most 5000 candidates can be compared. Reduce the range; no partial check was applied.')
    ids = [c.get('id') for c in items]
    if any((not x for x in ids)) or len(set(ids)) != len(ids):
        raise ValueError('Candidate identifiers are missing or duplicated. Regenerate or review the project.')
    started = time.monotonic()
    candidates = {c['id']: normalise_item(c) for c in items}
    originals = {c['id']: c for c in items}
    ordered = sorted(
        candidates.values(),
        key=lambda c: (
            {'preferred': 0, 'keep': 1}.get(c.get('review'), 2),
            ids.index(c['id'])
        )
    )
    representatives, groups, duplicates, skipped, ops = ([], [], [], [], {})
    for c in ordered:
        if time.monotonic() - started > 80:
            raise ValueError('Duplicate-check work limit exceeded. No partial decisions applied; reduce the comparison range.')
        if c.get('accepted') is False or c.get('review') == 'reject':
            skipped.append({
                'id': c['id'],
                'reason': 'candidate excluded by geometry checking or manual review'
            })
            continue
        _labels(c['input_parent'])
        witness, group_index = (None, None)
        for index, rep in enumerate(representatives):
            if Counter(rep['input_parent']['symbols']) != Counter(c['input_parent']['symbols']):
                continue
            if rep['id'] not in ops:
                ops[rep['id']] = _surface_ops(rep['input_parent'], opts)
            witness = compare_items(rep, c, ops[rep['id']], opts)
            if witness:
                group_index = index
                break
        if witness:
            original = originals[c['id']]
            duplicate = {
                'id': c['id'],
                'representative_id': rep['id'],
                **witness,
                'explanation': explain(witness, rep['input_parent']),
                'candidate_trace': copy.deepcopy(original.get('provenance', {})),
                'representative_trace': copy.deepcopy(originals[rep['id']].get('provenance', {})),
                'manual_reviewed': False
            }
            duplicates.append(duplicate)
            groups[group_index]['members'].append(c['id'])
        else:
            representatives.append(c)
            groups.append({'representative_id': c['id'], 'members': [c['id']]})
    return {
        'count': len(items),
        'unique_count': len(representatives),
        'groups': groups,
        'duplicates': duplicates,
        'skipped': skipped,
        'options': opts,
        'input_snapshot': snapshot(items),
        'scope_ids': ids,
        'manual_reviewed': False,
        'elapsed_s': time.monotonic() - started,
        'warning': 'Duplicate suggestions preserve originals and require review for false positives and missed matches.'
    }


def apply_review(items, report, selected_ids):
    scope = [next(
        (c for c in items if c['id'] == i),
        None
    ) for i in report['scope_ids']]
    if any((c is None for c in scope)) or snapshot(scope) != report['input_snapshot']:
        raise ValueError('Candidate structures or reaction information changed after checking. Repeat duplicate checking.')
    chosen = set(selected_ids)
    proposed = {x['id']: x for x in report['duplicates']}
    if not chosen <= set(proposed):
        raise ValueError('Cannot exclude a candidate absent from this preview.')
    result, undo = (copy.deepcopy(items), [])
    byid = {c['id']: c for c in result}
    for key in chosen:
        c, d = (byid[key], proposed[key])
        undo.append({
            'id': key,
            'fields': {k: copy.deepcopy(c[k]) for k in (
                'duplicateOf',
                'duplicateInfo',
                'checked'
            ) if k in c},
            'missing': [k for k in (
                'duplicateOf',
                'duplicateInfo',
                'checked'
            ) if k not in c]
        })
        c['duplicateOf'], c['duplicateInfo'], c['checked'] = (
            d['representative_id'],
            {**copy.deepcopy(d), 'manual_reviewed': True},
            False
        )
    return {'items': result, 'undo': undo}


def undo_review(items, undo):
    result = copy.deepcopy(items)
    byid = {c['id']: c for c in result}
    for entry in undo:
        c = byid.get(entry['id'])
        if c is None:
            continue
        for key in entry['missing']:
            c.pop(key, None)
        c.update(copy.deepcopy(entry['fields']))
    return result


def comparison_geometry(representative, candidate, witness):
    a, b = (normalise_item(representative), normalise_item(candidate))
    mapping, op = (witness['mapping'], witness['operation'])
    pa = _apply(a['input_parent']['positions'], op)
    pb = b['input_parent']['positions']
    shifts = []
    for i, j in enumerate(mapping):
        delta = sub(pa[i], pb[j])
        mic = displacement(pb[j], pa[i], a['input_parent'])
        shifts.append(sub(delta, mic))
    panels = {}
    for field in ('input_parent', 'structure'):
        left, right = (copy.deepcopy(a[field]), copy.deepcopy(b[field]))
        transformed = _apply(left['positions'], op)
        left['positions'] = [sub(x, t) for x, t in zip(transformed, shifts)]
        right['positions'] = [right['positions'][j] for j in mapping]
        right['symbols'], right['atom_ids'] = (left['symbols'][:], left['atom_ids'][:])
        right['adsorbate_indices'] = left.get('adsorbate_indices', [])[:]
        right['fixed_indices'] = left.get('fixed_indices', [])[:]
        if 'initial_magmoms' in left:
            right['initial_magmoms'] = copy.deepcopy(left['initial_magmoms'])
        panels[field] = {'representative': left, 'candidate': right}
    return panels
