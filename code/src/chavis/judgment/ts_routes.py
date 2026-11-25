from __future__ import annotations
from types import MappingProxyType
from .canonical_judgment import CRITERION_VERSION, evaluate
TS_ROUTE_POLICIES = MappingProxyType({
    'ni_drm': 'ni-drm-reaction-success-v1',
    'ni_legacy': 'proposed-v1',
    'generic_ts_current_legacy': 'automation-search-classification-current',
    'generic_ts_historical_legacy': 'automation-search-classification-legacy'
})


def evaluate_ts_route(
    route_id,
    saved_result,
    *,
    saved_IS=None,
    saved_FS=None,
    canonical_kwargs=None,
    native_classifier=None
):
    if route_id not in TS_ROUTE_POLICIES:
        raise ValueError('UNSUPPORTED_TS_JUDGMENT_ROUTE')
    if route_id == 'ni_drm':
        if saved_IS is None or saved_FS is None or canonical_kwargs is None:
            raise ValueError('CANONICAL_SAVED_EVIDENCE_REQUIRED')
        verdict = evaluate(saved_result, saved_IS, saved_FS, **canonical_kwargs)
        return {
            'route_id': route_id,
            'policy_version': CRITERION_VERSION,
            'verdict_schema': 'canonical_reaction_success',
            'canonical_primary_success': verdict.get('primary_success'),
            'route_verdict': verdict
        }
    if not callable(native_classifier):
        raise ValueError('NATIVE_ROUTE_CLASSIFIER_REQUIRED')
    native = native_classifier(saved_result)
    return {
        'route_id': route_id,
        'policy_version': TS_ROUTE_POLICIES[route_id],
        'verdict_schema': 'historical_route_native',
        'canonical_primary_success': None,
        'route_verdict': native
    }
