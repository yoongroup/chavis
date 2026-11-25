import math
POLICY_VERSION = 'saved-endpoint-return-v1'
WINDOW_RATIOS = (1.2, 2.0)
INTERVAL_APPROACH_A = 0.1
ENDPOINT_ADS_RMSD_A = 0.35
ENDPOINT_FREE_NI_RMSD_A = 0.25
ENDPOINT_BOND_TOLERANCE_A = 0.15
ENDPOINT_BOND_R0_RATIO = 0.1
ENDPOINT_APPROACH_A = 0.2
RAW_FREE_FMAX_EV_A = 0.03
MIN_CURVATURE_EV_A2 = 0.0
OBSERVATION_PERIOD = 5
FIRST_ELIGIBLE_STEP = 15
PERSISTENCE_CHECKS = 3


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def interval_distance(bond_A, r0_A):
    if not _finite(bond_A) or not _finite(r0_A) or bond_A < 0 or (r0_A <= 0):
        raise ValueError('Finite nonnegative bond and positive saved-IS r0 required')
    return max(WINDOW_RATIOS[0] * r0_A - bond_A, 0.0, bond_A - WINDOW_RATIOS[1] * r0_A)


def interval_progress(seed_bond_A, current_bond_A, r0_A):
    improvement = interval_distance(seed_bond_A, r0_A) - interval_distance(current_bond_A, r0_A)
    return dict(
        distance_seed_A=interval_distance(seed_bond_A, r0_A),
        distance_now_A=interval_distance(current_bond_A, r0_A),
        improvement_A=improvement,
        approached=improvement >= INTERVAL_APPROACH_A - 1e-12
    )


class EndpointReturnMonitor:

    def __init__(self, *, r0_A, context_key, mode='shadow'):
        if not _finite(r0_A) or r0_A <= 0:
            raise ValueError('Positive saved-IS r0 required')
        if not isinstance(context_key, str) or not context_key:
            raise ValueError('Nonempty frozen context key required')
        if mode not in ('shadow', 'enforce'):
            raise ValueError('Guard mode must be shadow or enforce')
        self.r0_A = float(r0_A)
        self.context_key = context_key
        self.mode = mode
        self.last_step = None
        self.streak_endpoint = None
        self.streak = 0

    def observe(
        self,
        *,
        step,
        context_key,
        endpoint_metrics,
        raw_free_fmax_eV_A,
        curvature_eV_A2,
        coordinates_saved
    ):
        base = dict(
            policy_version=POLICY_VERSION,
            mode=self.mode,
            step=step,
            streak=self.streak,
            endpoint=self.streak_endpoint
        )
        if type(step) is not int or step < 0 or step % OBSERVATION_PERIOD:
            raise ValueError('Optimizer step must be a nonnegative 5-step observation')
        if self.last_step is not None and step <= self.last_step:
            self.streak_endpoint, self.streak = (None, 0)
            return dict(
                base,
                code='GUARD_UNRESOLVED',
                reason='NONINCREASING_OPTIMIZER_STEP',
                streak=0,
                endpoint=None
            )
        consecutive = self.last_step is not None and step - self.last_step == OBSERVATION_PERIOD
        self.last_step = step
        if context_key != self.context_key or coordinates_saved is not True:
            self.streak_endpoint, self.streak = (None, 0)
            return dict(
                base,
                code='GUARD_UNRESOLVED',
                reason='CONTEXT_OR_COORDINATE_EVIDENCE_MISSING',
                streak=0,
                endpoint=None
            )
        if not _finite(raw_free_fmax_eV_A) or raw_free_fmax_eV_A < 0 or (not _finite(curvature_eV_A2)):
            self.streak_endpoint, self.streak = (None, 0)
            return dict(
                base,
                code='GUARD_UNRESOLVED',
                reason='PHYSICAL_FORCE_OR_CURVATURE_UNKNOWN',
                streak=0,
                endpoint=None
            )
        if not isinstance(endpoint_metrics, dict) or set(endpoint_metrics) != {'IS', 'FS'}:
            self.streak_endpoint, self.streak = (None, 0)
            return dict(
                base,
                code='GUARD_UNRESOLVED',
                reason='ENDPOINT_METRICS_MISSING',
                streak=0,
                endpoint=None
            )
        eligible = []
        signals = {}
        for label in ('IS', 'FS'):
            item = endpoint_metrics[label]
            names = (
                'ads_rmsd_A',
                'seed_ads_rmsd_A',
                'free_ni_rmsd_A',
                'bond_difference_A'
            )
            if not isinstance(item, dict) or any((not _finite(item.get(n)) or item[n] < 0 for n in names)):
                self.streak_endpoint, self.streak = (None, 0)
                return dict(
                    base,
                    code='GUARD_UNRESOLVED',
                    reason='ENDPOINT_METRICS_NONFINITE',
                    streak=0,
                    endpoint=None
                )
            improvement = item['seed_ads_rmsd_A'] - item['ads_rmsd_A']
            checks = dict(
                ads=item['ads_rmsd_A'] <= ENDPOINT_ADS_RMSD_A,
                free_ni=item['free_ni_rmsd_A'] <= ENDPOINT_FREE_NI_RMSD_A,
                bond=item['bond_difference_A'] <= max(
                    ENDPOINT_BOND_TOLERANCE_A,
                    ENDPOINT_BOND_R0_RATIO * self.r0_A
                ),
                improvement=improvement >= ENDPOINT_APPROACH_A - 1e-12,
                raw_force=raw_free_fmax_eV_A <= RAW_FREE_FMAX_EV_A,
                curvature=curvature_eV_A2 >= MIN_CURVATURE_EV_A2
            )
            signals[label] = dict(**checks, endpoint_approach_A=improvement)
            if all(checks.values()):
                eligible.append(label)
        if step < FIRST_ELIGIBLE_STEP:
            self.streak_endpoint, self.streak = (None, 0)
            return dict(
                base,
                code='OBSERVE_ONLY',
                reason='INITIAL_GRACE',
                signals=signals,
                streak=0,
                endpoint=None
            )
        chosen = min(
            eligible,
            key=lambda label: (endpoint_metrics[label]['ads_rmsd_A'], label)
        ) if eligible else None
        if chosen is None:
            self.streak_endpoint, self.streak = (None, 0)
            return dict(
                base,
                code='OBSERVE_ONLY',
                reason='RETURN_SIGNALS_INCOMPLETE',
                signals=signals,
                streak=0,
                endpoint=None
            )
        self.streak = self.streak + 1 if consecutive and self.streak_endpoint == chosen else 1
        self.streak_endpoint = chosen
        if self.streak < PERSISTENCE_CHECKS:
            return dict(
                base,
                code='OBSERVE_ONLY',
                reason='PERSISTENCE_PENDING',
                signals=signals,
                streak=self.streak,
                endpoint=chosen
            )
        return dict(
            base,
            code=('WOULD_ABORT_RETURN_' if self.mode == 'shadow' else 'ABORT_RETURN_') + chosen,
            reason='SUSTAINED_SAVED_ENDPOINT_RETURN',
            signals=signals,
            streak=self.streak,
            endpoint=chosen
        )
