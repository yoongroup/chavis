import math
from .policy import NUMERIC_POLICY, H_H_PAIR_POLICY_KEY, bond_intact, negative_curvature
from chavis.runtime.numeric_defaults import load_numeric_defaults
_FREQUENCY = load_numeric_defaults()['judgment']['native_automation_frequency']


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def classify_native(result, *, legacy=False):
    answer = _classify(result, legacy=legacy)
    if result.get('status') in ('running', 'interrupted', 'error') and answer['execution'] in ('error', 'incomplete'):
        probe = _classify(
            {**result, 'status': 'done', 'abort': None, 'connections': {}},
            legacy=legacy
        )
        if probe['candidate'] is True:
            answer.update(
                candidate=True,
                candidate_basis='CONVERGED_FULL_FREE_ONE_IMAG'
            )
    return answer


def classify(result, *, legacy=False):
    from chavis.judgment.ts_routes import evaluate_ts_route
    route = 'generic_ts_historical_legacy' if legacy else 'generic_ts_current_legacy'
    return evaluate_ts_route(
        route,
        result,
        native_classifier=lambda saved: classify_native(saved, legacy=legacy)
    )['route_verdict']


def _classify(result, *, legacy=False):
    state = result.get('status')
    if state in ('running', 'queued', 'interrupted') or state is None:
        return dict(
            execution='incomplete',
            candidate=None,
            verdict=None,
            reason='INCOMPLETE_EXECUTION'
        )
    if state == 'error':
        return dict(
            execution='error',
            candidate=None,
            verdict=None,
            reason='INFRASTRUCTURE_OR_UNRESOLVED_ERROR'
        )
    if state not in ('done', 'aborted'):
        return dict(
            execution='incomplete',
            candidate=None,
            verdict=None,
            reason='UNKNOWN_OR_CANCELLED_EXECUTION'
        )
    if state == 'aborted' and (not result.get('abort')):
        return dict(
            execution='error',
            candidate=None,
            verdict=None,
            reason='MISSING_ABORT_REASON'
        )
    if result.get('abort'):
        code = result['abort'].get('code')
        if code in {'ABORT_RETURN_IS', 'ABORT_RETURN_FS'}:
            return dict(
                execution='completed',
                candidate=False,
                verdict=code,
                reason=code,
                category='OPERATIONAL_ENDPOINT_RETURN_NOT_CHEMICAL_FAILURE'
            )
        if code not in {
            'ABORT_DIVERGE',
            'ABORT_SLAB',
            'ABORT_BELOW_SURFACE',
            'ABORT_DRIFT',
            'ABORT_FLOAT'
        }:
            return dict(
                execution='error',
                candidate=None,
                verdict=None,
                reason='INVALID_ABORT_CODE'
            )
        return dict(execution='completed', candidate=False, verdict=code, reason=code)

    def out(verdict, candidate=None):
        return dict(
            execution='completed',
            candidate=candidate,
            verdict=verdict,
            reason=verdict
        )
    dm = result.get('dimer')
    if not dm or not all((finite(dm.get(k)) for k in ('fmax', 'curvature'))):
        return out('INCOMPLETE_DIMER')
    if dm.get('converged') is not True or dm['fmax'] > NUMERIC_POLICY['ts_fmax_eV_A'] or dm['fmax'] < 0 or (not negative_curvature(dm['curvature'])):
        return out('NOT_CONVERGED', False)
    if result.get('ts_judge') != 'OK':
        return out(
            'PHYSICS_REJECT' if result.get('ts_judge') else 'MISSING_TS_JUDGE',
            False
        )
    fr = result.get('frequencies')
    if not fr:
        return out('NO_FREQUENCIES')
    nf, modes = (fr.get('n_free'), fr.get('cm1'))
    if type(nf) is not int or nf <= 0 or (not isinstance(modes, list)) or (len(modes) != nf * 3) or (not all((finite(x) for x in modes))):
        return out('INCOMPLETE_SPECTRUM')
    if not legacy and (fr.get('completed') is not True or fr.get('delta_A') != _FREQUENCY['delta_A'] or fr.get('nfree') != _FREQUENCY['nfree'] or (len(set(fr.get('indices', []))) != nf) or (fr.get('indices') != fr.get('expected_free_indices'))):
        return out('INCOMPLETE_SPECTRUM_METADATA')
    if not legacy:
        ts = result.get('TS', {})
        if not isinstance(ts, dict):
            return out('INCOMPLETE_TS_MASK')
        fixed = ts.get('fixed_indices')
        symbols = ts.get('symbols', [])
        if not isinstance(fixed, list) or not symbols or len(set(fixed)) != len(fixed) or any((type(i) is not int or not 0 <= i < len(symbols) for i in fixed)):
            return out('INCOMPLETE_TS_MASK')
        expected_free = [i for i in range(len(symbols)) if i not in fixed]
        if fr.get('indices') != expected_free or nf != len(expected_free):
            return out('PARTIAL_HESSIAN_NOT_FULL_FREE')
    imag = [
        i for i,
        x in enumerate(modes) if x < -NUMERIC_POLICY['imaginary_epsilon_cm1']
    ]
    if fr.get('imaginary_modes') != imag or fr.get('imaginary_count') != len(imag):
        return out('INCONSISTENT_SPECTRUM')
    if len(imag) != 1:
        return out('MULTI_IMAG' if imag else 'ZERO_IMAG', False)
    con = result.get('connections', {})
    if any((not isinstance(con.get(s), dict) for s in ('minus', 'plus'))):
        return out('NO_CONNECTIONS', True)
    if any((con[s].get('converged') is not True for s in ('minus', 'plus'))):
        return out('CONNECTION_NOT_CONVERGED', True)
    if any((con[s].get('judge') != 'OK' for s in ('minus', 'plus'))):
        return out('PHYSICS_REJECT', True)
    if any((not finite(con[s].get('d_target_A')) or con[s]['d_target_A'] <= 0 for s in ('minus', 'plus'))):
        return out('INCOMPLETE_CONNECTION', True)
    if not legacy:
        d_is = result.get('d_IS_A')
        if not finite(d_is) or d_is <= 0 or result.get('target_bond_intact_ratio') != NUMERIC_POLICY['reference_bond_intact_ratio'] or any((not finite(con[s].get('fmax')) or not 0 <= con[s]['fmax'] <= NUMERIC_POLICY['endpoint_fmax_eV_A'] for s in ('minus', 'plus'))):
            return out('INCOMPLETE_CONNECTION_METADATA', True)
        pair = result.get('target_bond_elements')
        if pair == ['H', 'H'] and result.get('h_h_pair_policy_key') != H_H_PAIR_POLICY_KEY:
            return out('INCOMPLETE_CONNECTION_METADATA', True)
        if any((con[s].get('target_state') != ('intact' if (bond_intact(
            con[s]['d_target_A'],
            pair[0],
            pair[1],
            reference_length_A=d_is
        ) if pair == ['H', 'H'] else con[s]['d_target_A'] <= NUMERIC_POLICY['reference_bond_intact_ratio'] * d_is) else 'broken') for s in ('minus', 'plus'))):
            return out('INCONSISTENT_TARGET_STATE', True)
    if {con[s].get('target_state') for s in ('minus', 'plus')} != {'intact', 'broken'}:
        return out('OTHER_CONNECTION', True)
    return out('TARGET_TS', True)
