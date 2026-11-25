from __future__ import annotations
from types import MappingProxyType
from chavis.runtime.numeric_defaults import load_numeric_defaults
_DATA = load_numeric_defaults()['judgment']
NUMERIC_POLICY_VERSION = _DATA['version']
NUMERIC_POLICY = MappingProxyType(_DATA['rules'])
RADII = MappingProxyType(_DATA['radii_A'])
V1_PROFILE = 'v1'
V2_PROFILE = 'v2'
V3_PROFILE = 'v3'
FS_RELAXATION_TRIAL_CAP_DEFAULT = 5
REACTION_ROLE_POLICY_KEY = 'approved_primary15_requested_parent_and_star_product_roles_v3'
APPROVED_PRIMARY_REACTIONS = frozenset((
    'D01',
    'D02',
    'D03',
    'D04',
    'D05',
    'D06',
    'D07',
    'D08',
    'D09',
    'D10',
    'D13',
    'N01',
    'N02',
    'N03',
    'N04'
))
PARENT_ROLE_OVERRIDES = MappingProxyType({
    'D01': ('detached_molecule', 'weak_contact'),
    'D05': ('surface_contact', 'weak_contact'),
    'D13': ('surface_contact', 'weak_contact', 'detached_molecule')
})


def reaction_role_contract(
    reaction_id,
    *,
    declared=None,
    equation=None,
    requested_direction='dissociation'
):
    if reaction_id not in APPROVED_PRIMARY_REACTIONS:
        raise ValueError('UNSUPPORTED_REACTION_ROLE_CONTRACT: ' + str(reaction_id))
    allowed = PARENT_ROLE_OVERRIDES.get(
        reaction_id,
        ('surface_contact', 'weak_contact')
    )
    if declared is not None and tuple(declared) != allowed:
        raise ValueError('DECLARED_PARENT_ROLES_DIFFER_FROM_APPROVED_CONTRACT')
    if equation is not None:
        sides = equation.split('->') if isinstance(equation, str) else []
        if len(sides) != 2 or requested_direction not in ('dissociation', 'association'):
            raise ValueError('INVALID_REQUESTED_REACTION_EQUATION')
        product_side = sides[0] if requested_direction == 'association' else sides[1]
        products = [part.strip() for part in product_side.split('+')]
        if not products or any((not part.endswith('*') for part in products)):
            raise ValueError('REQUESTED_PRODUCT_FRAGMENT_LACKS_STAR_ROLE')
    return {
        'reaction_id': reaction_id,
        'policy_key': REACTION_ROLE_POLICY_KEY,
        'allowed_parent_roles': list(allowed),
        'required_product_role': 'surface_contact',
        'product_role_source': 'saved_requested_equation_star_products' if equation is not None else 'approved_primary_reaction_equations_star_products',
        'detached_fallback_separate_channel': reaction_id == 'D05'
    }


H_H_GAS_REFERENCE_LENGTH_A = 0.7411889271385169
H_H_PAIR_POLICY_KEY = 'same_pinned_uma_omol_relaxed_h2_length_times_reference_bond_intact_ratio'
H_H_PAIR_POLICY = MappingProxyType({
    'key': H_H_PAIR_POLICY_KEY,
    'reference_length_A': H_H_GAS_REFERENCE_LENGTH_A,
    'reference_bond_intact_ratio': NUMERIC_POLICY['reference_bond_intact_ratio'],
    'model': 'uma-s-1p2p1',
    'model_revision': 'f611b917d9c68566bbbeccbb0aa0f7cad1696cb2',
    'model_task': 'omol',
    'source': 'reference/Ni100/preparation/items/D13/OMOL_gas_parent/gas_optimized.json'
})


def bond_threshold(
    symbol_a,
    symbol_b,
    *,
    reference_length_A=None,
    nonreference_scale=None,
    criterion_profile=V2_PROFILE
):
    if criterion_profile not in (V1_PROFILE, V2_PROFILE, V3_PROFILE):
        raise ValueError('UNSUPPORTED_JUDGMENT_PROFILE')
    if criterion_profile != V1_PROFILE and symbol_a == symbol_b == 'H':
        return H_H_GAS_REFERENCE_LENGTH_A * NUMERIC_POLICY['reference_bond_intact_ratio']
    if reference_length_A is not None:
        return float(reference_length_A) * NUMERIC_POLICY['reference_bond_intact_ratio']
    scale = NUMERIC_POLICY['nonreference_bond_radius_scale'] if nonreference_scale is None else float(nonreference_scale)
    return scale * (RADII[symbol_a] + RADII[symbol_b])


def bond_intact(
    distance_A,
    symbol_a,
    symbol_b,
    *,
    reference_length_A=None,
    nonreference_scale=None,
    criterion_profile=V2_PROFILE,
    inclusive=True
):
    threshold = bond_threshold(
        symbol_a,
        symbol_b,
        reference_length_A=reference_length_A,
        nonreference_scale=nonreference_scale,
        criterion_profile=criterion_profile
    )
    if criterion_profile != V1_PROFILE and symbol_a == symbol_b == 'H':
        return 0 < distance_A < threshold
    return 0 < distance_A <= threshold if inclusive else 0 < distance_A < threshold


def negative_curvature(curvature):
    return curvature < NUMERIC_POLICY['curvature_boundary_eV_A2']


def surface_role(
    scaled_ni_radius_ratio,
    min_height_A,
    *,
    criterion_profile=V1_PROFILE,
    intact_parent=False
):
    """Classify distance divided by the actual adsorbate/metal radii sum.

    ``scaled_ni_radius_ratio`` and the ``*_Ni_radius_scale`` JSON keys retain
    their historical names for compatibility. They are dimensionless ratios
    for the selected metal, including Cu and Rh; they never select a Ni radius.
    """
    if NUMERIC_POLICY['boundary_ambiguity_interval_enabled']:
        interval = NUMERIC_POLICY['boundary_ambiguity_ratio_interval']
        if interval is None or len(interval) != 2 or interval[0] > interval[1]:
            raise ValueError('BOUNDARY_AMBIGUITY_INTERVAL_MISSING_OR_INVALID')
        if interval[0] <= scaled_ni_radius_ratio <= interval[1]:
            return 'ambiguous'
    if scaled_ni_radius_ratio <= NUMERIC_POLICY['near_Ni_radius_scale']:
        return 'surface_contact'
    if scaled_ni_radius_ratio >= NUMERIC_POLICY['detached_Ni_radius_scale'] and min_height_A >= NUMERIC_POLICY['detached_min_height_A']:
        return 'detached_molecule'
    if criterion_profile in (V2_PROFILE, V3_PROFILE) and intact_parent and (NUMERIC_POLICY['near_Ni_radius_scale'] < scaled_ni_radius_ratio < NUMERIC_POLICY['detached_Ni_radius_scale']):
        return 'weak_contact'
    return 'ambiguous'


def legacy_rules():
    return dict(
        reference_bond_intact_ratio=NUMERIC_POLICY['reference_bond_intact_ratio'],
        nonreference_bond_radius_scale=NUMERIC_POLICY['nonreference_bond_radius_scale'],
        near_Ni_radius_scale=NUMERIC_POLICY['near_Ni_radius_scale'],
        detached_Ni_radius_scale=NUMERIC_POLICY['detached_Ni_radius_scale'],
        detached_min_height_A=NUMERIC_POLICY['detached_min_height_A'],
        endpoint_fmax_eV_A=NUMERIC_POLICY['endpoint_fmax_eV_A'],
        ts_fmax_eV_A=NUMERIC_POLICY['ts_fmax_eV_A'],
        imaginary_epsilon_cm1=NUMERIC_POLICY['imaginary_epsilon_cm1'],
        role_evidence_level='GEOMETRIC_SCREENING_NOT_GAS_BOUNDARY_PROOF'
    )
