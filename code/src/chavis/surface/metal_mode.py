from __future__ import annotations
import numpy as np
LAYER_TOL_A = 0.35


def layers(slab_record):
    st = slab_record['structure']
    if st.get('layers'):
        return list(st['layers'])
    z = np.asarray(st['positions'], float)[:, 2]
    order = np.argsort(-z)
    out = [0] * len(z)
    layer = 0
    last = None
    for i in order:
        if last is None or last - z[i] > LAYER_TOL_A:
            layer += 1
            last = z[i]
        out[int(i)] = layer
    return out


def parse(spec):
    if spec in ('all', None):
        return 'all'
    if spec in ('none', '', []):
        return []
    if isinstance(spec, str):
        spec = [s for s in spec.replace(' ', '').split(',') if s]
    vals = sorted({int(x) for x in spec})
    if any((v < 1 for v in vals)):
        raise ValueError('layer numbers start at 1 (top layer)')
    return vals


def fixed_indices(slab_record, fix_layers='all'):
    lay = layers(slab_record)
    spec = parse(fix_layers)
    if spec == 'all':
        return list(range(len(lay)))
    missing = set(spec) - set(lay)
    if missing:
        raise ValueError(f'slab has layers 1..{max(lay)}; no layer {sorted(missing)}')
    return [i for i, l in enumerate(lay) if l in spec]


def describe(slab_record, fix_layers):
    lay = layers(slab_record)
    spec = parse(fix_layers)
    fixed = sorted(set(range(1, max(lay) + 1)) if spec == 'all' else set(spec))
    return {
        'n_layers': max(lay),
        'fixed_layers': fixed,
        'free_layers': [l for l in range(1, max(lay) + 1) if l not in fixed],
        'numbering': '1 = exposed top layer'
    }
