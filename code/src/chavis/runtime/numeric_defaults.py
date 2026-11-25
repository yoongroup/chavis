from __future__ import annotations
from functools import lru_cache
from .hashing import sha256_file
import json
import math
from pathlib import Path
NUMERIC_DEFAULTS_PATH = Path(__file__).with_name('numeric_defaults.json')


@lru_cache(maxsize=1)
def load_numeric_defaults():
    with NUMERIC_DEFAULTS_PATH.open('rb') as stream:
        data = json.load(stream)
    if data.get('schema') != 'chavis_shared_numeric_defaults_v1':
        raise ValueError('UNSUPPORTED_SHARED_NUMERIC_DEFAULTS')
    judgment = data.get('judgment', {})
    search = data.get('search', {})
    inputs = data.get('inputs', {})
    rules = judgment.get('rules', {})
    routes = search.get('idpp_routes', {})
    required_rules = (
        'reference_bond_intact_ratio',
        'nonreference_bond_radius_scale',
        'reference_input_radius_scale',
        'near_Ni_radius_scale',
        'detached_Ni_radius_scale',
        'detached_min_height_A',
        'endpoint_fmax_eV_A',
        'ts_fmax_eV_A',
        'imaginary_epsilon_cm1',
        'curvature_boundary_eV_A2',
        'cell_context_atol_A',
        'fixed_position_context_atol_A'
    )
    if not judgment.get('version') or any((type(rules.get(k)) not in (int, float) or not math.isfinite(rules[k]) for k in required_rules)) or rules.get('curvature_comparison') != 'strictly_less_than' or (type(rules.get('boundary_ambiguity_interval_enabled')) is not bool) or (rules['boundary_ambiguity_interval_enabled'] and rules.get('boundary_ambiguity_ratio_interval') is None) or (not isinstance(judgment.get('radii_A'), dict)) or (not all((type(v) in (int, float) and math.isfinite(v) and (v > 0) for v in judgment['radii_A'].values()))) or (not isinstance(routes, dict)) or (not routes):
        raise ValueError('INVALID_SHARED_NUMERIC_DEFAULTS')
    for name, route in routes.items():
        if not name or type(route.get('mic')) is not bool or type(route.get('fragment_consistent_images')) is not bool or (type(route.get('steps')) is not int) or (route['steps'] <= 0) or (type(route.get('fmax_eV_A')) not in (int, float)) or (not math.isfinite(route['fmax_eV_A'])) or (route['fmax_eV_A'] <= 0):
            raise ValueError('INVALID_SHARED_IDPP_ROUTE')
    if type(inputs.get('embed_seed')) is not int or type(inputs.get('embed_tries')) is not int or inputs['embed_tries'] <= 0 or (type(inputs.get('gas_fmax')) not in (int, float)) or (not math.isfinite(inputs['gas_fmax'])) or (inputs['gas_fmax'] <= 0) or (type(inputs.get('gas_steps')) is not int) or (inputs['gas_steps'] <= 0) or (not isinstance(inputs.get('spin_override'), dict)) or (not all((type(v) is int for v in inputs['spin_override'].values()))) or (not isinstance(inputs.get('placeholder'), str)) or (not inputs['placeholder']):
        raise ValueError('INVALID_SHARED_INPUT_DEFAULTS')
    gui = search.get('gui_ts_defaults')
    gui_stage = search.get('gui_ts_stage_constants')
    native_frequency = judgment.get('native_automation_frequency')
    gui_numeric = (
        'images',
        'endpoint_fmax',
        'neb_fmax',
        'ts_fmax',
        'endpoint_steps',
        'neb_steps',
        'dimer_steps',
        'delta',
        'connection_displacement',
        'match_rms',
        'max_fs_trials',
        'rmin',
        'rmax',
        'half_angle'
    )
    stage_numeric = (
        'endpoint_bfgs_maxstep_A',
        'neb_fire_dt',
        'neb_fire_maxstep_A',
        'warm_neb_fmax_eV_A',
        'fs_collision_scale',
        'accelerated_inference_parity_tolerance',
        'mode_projection_min_abs'
    )
    if not isinstance(gui, dict) or not isinstance(gui.get('species'), str) or (not gui['species']) or (not isinstance(gui.get('site'), str)) or (not gui['site']) or any((type(gui.get(key)) not in (int, float) or not math.isfinite(gui[key]) or gui[key] <= 0 for key in gui_numeric)) or (not isinstance(gui_stage, dict)) or any((type(gui_stage.get(key)) not in (int, float) or not math.isfinite(gui_stage[key]) or gui_stage[key] <= 0 for key in stage_numeric)) or (not isinstance(native_frequency, dict)) or (type(native_frequency.get('delta_A')) not in (int, float)) or (not math.isfinite(native_frequency['delta_A'])) or (native_frequency['delta_A'] <= 0) or (type(native_frequency.get('nfree')) is not int) or (native_frequency['nfree'] <= 0):
        raise ValueError('INVALID_SHARED_GUI_OR_NATIVE_NUMERIC_DEFAULTS')
    data['source'] = {
        'package_relative_path': 'chavis/runtime/numeric_defaults.json',
        'sha256': sha256_file(NUMERIC_DEFAULTS_PATH)
    }
    return data
