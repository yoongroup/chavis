from __future__ import annotations
import contextlib, copy, io, json, sys, traceback
from pathlib import Path
from chavis_bootstrap import ensure_chavis_import_path
ensure_chavis_import_path()
from geometry import validate, interpolate, compatible, to_ase, from_ase
IDPP_ROUTE_POLICY = 'normal_reaction_local_mic_idpp_v1'


def main(request):
    if str(request.get('task', '')).startswith('algorithm_'):
        from algorithms.tasks import execute
        return execute(request)
    if request.get('params', {}).get('mapping_alternatives'):
        from algorithms.tasks import run_mappings
        return run_mappings(request, main)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        if request.get('task') == 'dedup':
            from duplicate_filter import review_duplicates
            result = review_duplicates(
                request.get('items', []),
                request.get('options', {})
            )
        elif request.get('task') == 'idpp':
            from chavis.search.mic_idpp import prepare_idpp, PathPreparationError
            from chavis.search.policy import NORMAL_REACTION_MIC_IDPP
            from chavis.runtime.numeric_policy import load_effective_numeric_policy
            try:
                frames, diagnostics = prepare_idpp(
                    request['input'],
                    request['output'],
                    int(request.get('images', 8)),
                    policy=NORMAL_REACTION_MIC_IDPP,
                    path_class=request.get('path_class', 'local_reaction')
                )
            except PathPreparationError as exc:
                raise ValueError(f'{exc}: {json.dumps(
                    exc.diagnostics,
                    ensure_ascii=False,
                    allow_nan=False
                )}') from exc
            result = {
                'frames': frames,
                'diagnostics': diagnostics,
                'effective_numeric_policy': load_effective_numeric_policy(NORMAL_REACTION_MIC_IDPP.name),
                'warning': 'Local MIC/fragment-consistent IDPP initial guess only; review periodic image shifts and collision diagnostics. Deliberate long diffusion is unsupported. No TS/MEP or physical energies computed.'
            }
        else:
            code = request['code']
            namespace = {
                '__name__': 'user_generator',
                '__file__': '<editor>',
                '__builtins__': __builtins__
            }
            exec(compile(code, '<generator editor>', 'exec'), namespace)
            fun = namespace.get('generate')
            if not callable(fun):
                raise ValueError('Define generate(parent, params). It must return a list of candidate dictionaries.')
            values = fun(
                copy.deepcopy(request['parent']),
                copy.deepcopy(request.get('params', {}))
            )
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError('generate() must return a non-empty list of candidates.')
            limit = 10000 if request.get('params', {}).get('algorithm_mode') or request.get('params', {}).get('rotations_deg') else 300
            if len(values) > limit:
                raise ValueError(f'Preview limit: {limit} candidates per run.')
            candidates = []
            for i, v in enumerate(values):
                if 'structure' not in v:
                    v = {'structure': v}
                v = copy.deepcopy(v)
                v['structure'] = validate(v['structure'])
                if 'input_parent' in v:
                    v['input_parent'] = validate(v['input_parent'])
                v.setdefault('name', f'candidate_{i + 1:03d}.extxyz')
                v.setdefault('accepted', True)
                v.setdefault('reason', '')
                candidates.append(v)
            result = {'candidates': candidates}
    result['log'] = stream.getvalue()[-20000:]
    return result


if __name__ == '__main__':
    destination = Path(sys.argv[2])
    try:
        request = json.loads(Path(sys.argv[1]).read_text())
        result = main(request)
        destination.write_text(json.dumps(
            {'ok': True, **result},
            ensure_ascii=False,
            allow_nan=False
        ))
    except BaseException as error:
        destination.write_text(json.dumps(
            {'ok': False, 'error': str(error), 'traceback': traceback.format_exc()},
            ensure_ascii=False
        ))
        sys.exit(1)
