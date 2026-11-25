from __future__ import annotations
import contextlib
import copy
import io
from geometry import validate
from .generation import map_recipe, prepare_pair, signature
from .review import review, explain, snapshot
from duplicate_filter import review_duplicates


def review_items(items, options=None):
    opts = {'permutation': True, 'pair': True, **(options or {})}
    if opts['pair']:
        return review(items, opts)
    result = review_duplicates(items, opts)
    sources = {x['id']: x for x in items}
    for d in result['duplicates']:
        d['explanation'] = explain(
            d,
            validate(sources[d['representative_id']]['structure'])
        ).replace(
            'use the same correspondence and transform for parent and separated structures',
            'compare selected individual structures; reaction pairs are not checked'
        )
    result['scope_ids'] = [x['id'] for x in items]
    result['input_snapshot'] = snapshot(items)
    return result


def execute(request):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        task = request['task']
        if task == 'algorithm_map':
            result = map_recipe(request['source'], request['target'], request['params'])
        elif task == 'algorithm_pair':
            result = prepare_pair(
                request['reactant'],
                request['product'],
                request.get('params'),
                request.get('manual_parent_side')
            )
        elif task == 'algorithm_dedup':
            result = review_items(request.get('items', []), request.get('options'))
        else:
            raise ValueError('Unknown algorithm task')
    result['log'] = stream.getvalue()[-20000:]
    return result


def run_mappings(request, original_main):
    p = request.get('params', {})
    choices = p.get('mapping_alternatives')
    if not isinstance(choices, list) or not 1 <= len(choices) <= 256:
        raise ValueError('Atom correspondence list is missing or exceeds 256 entries.')
    if p.get('mapping_target_signature') != signature(request['parent']):
        raise ValueError('Target parent changed after correspondence review. Recheck correspondence.')
    out, logs = ([], [])
    for index, choice in enumerate(choices, 1):
        req = copy.deepcopy(request)
        atom_fields = (
            'adsorbate',
            'moving',
            'anchor',
            'surface_indices',
            'reaction_edits',
            'generation_edits',
            'bond_overrides',
            'atom_mapping_source',
            'mapping_alternative'
        )
        req['params'] = {**p, **{k: v for k, v in choice.items() if k in atom_fields}}
        req['params'].pop('mapping_alternatives', None)
        req['params'].pop('mapping_target_signature', None)
        result = original_main(req)
        logs.append(f'Atom correspondence {index}/{len(choices)}\n' + result.get('log', ''))
        for c in result['candidates']:
            c.setdefault('provenance', {})['mapping_alternative'] = index
            out.append(c)
        if len(out) > 10000:
            raise ValueError('Candidates across correspondences exceed 10000. Partial results were not returned.')
    return {'candidates': out, 'log': '\n'.join(logs)[-20000:]}
