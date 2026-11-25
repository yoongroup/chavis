from __future__ import annotations
from dataclasses import asdict
from .hashing import sha256_file
import json
from pathlib import Path
from chavis.judgment.policy import NUMERIC_POLICY, NUMERIC_POLICY_VERSION
from chavis.search.policy import NI_DRM_IDPP, NORMAL_REACTION_MIC_IDPP
from .numeric_defaults import load_numeric_defaults


def _digest(path):
    return sha256_file(path)


def load_effective_numeric_policy(route, campaign_policy_path=None):
    choices = {
        NORMAL_REACTION_MIC_IDPP.name: NORMAL_REACTION_MIC_IDPP,
        NI_DRM_IDPP.name: NI_DRM_IDPP
    }
    if route not in choices:
        raise ValueError('UNSUPPORTED_NUMERIC_ROUTE_OR_LONG_DIFFUSION')
    idpp = choices[route]
    defaults = load_numeric_defaults()
    campaign = None
    source = None
    if campaign_policy_path is not None:
        path = Path(campaign_policy_path).resolve(strict=True)
        campaign = json.loads(path.read_text(encoding='utf-8'))
        direct = campaign.get('direct', {})
        if direct.get('idpp_mic') is not idpp.mic or direct.get('idpp_fire_max_steps') != idpp.steps or direct.get('idpp_surrogate_fmax') != idpp.fmax_eV_A:
            raise ValueError('SAVED_CAMPAIGN_IDPP_POLICY_MISMATCH')
        source = {'resolved_path': str(path), 'sha256': _digest(path)}
    return {
        'schema': 'chavis_effective_numeric_policy_v1',
        'route': route,
        'declarative_defaults_source': defaults['source'],
        'judgment': {
            'version': NUMERIC_POLICY_VERSION,
            'applicability': 'Ni36_complete_saved_canonical_evidence_only',
            'active_for_this_operation': True,
            'native_automation_frequency': defaults['judgment']['native_automation_frequency'],
            **dict(NUMERIC_POLICY)
        },
        'search': {
            'idpp': asdict(idpp),
            'supported_path_class': 'local_reaction',
            'long_diffusion_opt_in_supported': False,
            'gui_ts_defaults': defaults['search']['gui_ts_defaults'],
            'gui_ts_stage_constants': defaults['search']['gui_ts_stage_constants']
        },
        'inputs': defaults['inputs'],
        'campaign_policy_source': source,
        'campaign_policy': campaign
    }
