from __future__ import annotations
from dataclasses import dataclass
from chavis.runtime.numeric_defaults import load_numeric_defaults


@dataclass(frozen=True)
class IDPPPolicy:
    name: str
    mic: bool
    fragment_consistent_images: bool
    steps: int
    fmax_eV_A: float


_ROUTES = load_numeric_defaults()['search']['idpp_routes']


def _idpp_policy(name):
    return IDPPPolicy(name=name, **_ROUTES[name])


LEGACY_GUI_IDPP = _idpp_policy('legacy_gui_cartesian_idpp_v1')
NI_DRM_IDPP = _idpp_policy('ni_drm_mic_idpp_v1')
NORMAL_REACTION_MIC_IDPP = _idpp_policy('normal_reaction_local_mic_idpp_v1')


@dataclass(frozen=True)
class AutomationPolicy:
    name: str
    allow_partial_scan_resume: bool
    default_endpoint_guard_mode: str | None
    include_resume_provenance: bool
    run_meta_note: str


LEGACY_GUI_AUTOMATION = AutomationPolicy(
    name='legacy_gui_automation_v1',
    allow_partial_scan_resume=False,
    default_endpoint_guard_mode=None,
    include_resume_provenance=False,
    run_meta_note='Injected model context'
)
NI_DRM_AUTOMATION = AutomationPolicy(
    name='ni_drm_automation_v1',
    allow_partial_scan_resume=True,
    default_endpoint_guard_mode='enforce',
    include_resume_provenance=True,
    run_meta_note='Sequential campaign; injected model settings are recorded in model_context'
)
