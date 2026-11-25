import math
from copy import deepcopy
SCHEMA = 'chavis_ts_cost_v1'


def _number(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: a measured number is required')
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError(f'{name}: invalid measurement')
    return value


def build_ledger(result, attempt_id):
    if not attempt_id or result.get('mode') not in ('fast', 'slow'):
        raise ValueError('attempt_id and fast/slow result required')
    rows = [deepcopy(s) for r in result['routes'] for s in r['seeds']]
    rows.sort(key=lambda s: s['order'])
    seen = set()
    previous_calls = previous_seconds = 0
    totals = {k: {
        'calls': 0,
        'seconds': 0.0,
        'complete': True
    } for k in ('dimer', 'freq', 'conn')}
    first = None
    candidates = []
    for order, row in enumerate(rows, 1):
        sid = row['seed_id']
        if not isinstance(sid, str) or not sid or sid in seen or (row['order'] != order):
            raise ValueError('duplicate seed or non-contiguous candidate order')
        seen.add(sid)
        cp = row['checkpoint']
        calls = _number(cp['calls_through'], 'checkpoint.calls', True)
        seconds = _number(cp['seconds_through'], 'checkpoint.seconds')
        if calls < previous_calls or seconds < previous_seconds:
            raise ValueError('candidate checkpoint decreased')
        previous_calls, previous_seconds = (calls, seconds)
        judged = row['judged']
        raw = judged.get('share_raw')
        confident = judged.get('validated') is True and isinstance(raw, (int, float)) and (not isinstance(raw, bool)) and math.isfinite(raw) and (raw >= result['fast_stop_share'])
        if confident and first is None:
            first = {'seed_id': sid, 'order': order, **cp}
        cost = row.get('cost', {})
        for stage, total in totals.items():
            entry = cost.get(stage)
            if entry is None:
                total['complete'] = False
                continue
            if entry.get('status') not in ('ran', 'error', 'skipped', 'unknown'):
                raise ValueError('unknown cost status')
            if entry['status'] in ('skipped', 'unknown') and (not entry.get('reason')):
                raise ValueError('skipped/unknown cost requires reason')
            if entry['status'] == 'skipped' and (entry.get('calls') != 0 or entry.get('seconds') != 0):
                raise ValueError('skipped stage has nonzero cost')
            for key in ('calls', 'seconds'):
                value = entry.get(key)
                if value is None:
                    total['complete'] = False
                else:
                    total[key] += _number(value, f'{stage}.{key}', key == 'calls')
        candidates.append({k: row.get(k) for k in (
            'seed_id',
            'order',
            'stage',
            'judged',
            'cost',
            'checkpoint'
        )})
    if first != result.get('fast_stop'):
        raise ValueError('fast_stop differs from first raw-share-qualified checkpoint')
    if result['mode'] == 'fast' and first and (first['order'] != len(rows)):
        raise ValueError('fast result contains candidates after stopping point')
    pair_calls = _number(result['model_calls'], 'pair.calls', True)
    pair_seconds = _number(result['seconds'], 'pair.seconds')
    if previous_calls > pair_calls or previous_seconds > pair_seconds + 0.051:
        raise ValueError('checkpoint exceeds enclosing pair measurement')
    if sum((v['calls'] for v in totals.values())) > pair_calls:
        raise ValueError('candidate costs exceed pair calls (possible double count)')
    return {
        'schema': SCHEMA,
        'attempt_id': str(attempt_id),
        'mode': result['mode'],
        **({
            'route_plan': result['route_plan'],
            'effective_route_plan': result['effective_route_plan']
        } if 'effective_route_plan' in result else {}),
        'pair_total': {'calls': pair_calls, 'seconds': pair_seconds},
        'stage_breakdown': totals,
        'candidates': candidates,
        'fast_prefix': first,
        'validated_seed_ids': [r['seed_id'] for r in rows if r['judged'].get('validated') is True],
        'fallback_used': first is None and any((r['judged'].get('validated') is True for r in rows)),
        'accounting': 'pair_total includes preparation and stages; never sum total with breakdown',
        'prefix_interpretation': 'observed slow/fast pair prefix, not independent fast wall-time savings',
        'model_load': 'outside pair; see runner attempt record'
    }
