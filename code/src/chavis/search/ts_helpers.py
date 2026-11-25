from __future__ import annotations
import numpy as np
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from .geometry import RADII, from_ase
from .mic_idpp import prepare_idpp
from .policy import IDPPPolicy, NORMAL_REACTION_MIC_IDPP
from chavis.runtime.numeric_defaults import load_numeric_defaults
from chavis.judgment.policy import V2_PROFILE, bond_intact
_BOND_SCALE = load_numeric_defaults()['search']['local_geometry']['surviving_bond_radius_scale']


def free_indices(atoms):
    fixed = set()
    for con in atoms.constraints:
        if not isinstance(con, FixAtoms):
            raise ValueError('Only fixed-atom constraints supported')
        fixed.update((int(i) for i in con.get_indices()))
    return [i for i in range(len(atoms)) if i not in fixed]


def _assert_saved_surface_profile(atoms):
    profile = atoms.info.get('surface_profile')
    if not profile:
        return None
    n = profile['n_slab']
    fixed = sorted(set(range(len(atoms))) - set(free_indices(atoms)))
    if fixed != profile['fixed_indices']:
        raise ValueError('Surface constraints differ from the stored constraints.')
    if any((s != profile['element'] for s in atoms.get_chemical_symbols()[:n])):
        raise ValueError('Surface metal or atom order mismatch')
    if atoms.info.get('adsorbate_indices') != list(range(n, len(atoms))):
        raise ValueError('Slab/adsorbate atom correspondence mismatch')
    if not np.array_equal(atoms.cell, profile['cell']) or not np.array_equal(atoms.pbc, profile['pbc']):
        raise ValueError('Stored surface cell/PBC mismatch')
    if profile.get('facet') == 'fcc100':
        from chavis.judgment.surface_context import selected_surface_element
        if selected_surface_element(from_ase(atoms)) != 'Ni':
            raise ValueError('Validated Ni(100) surface profile required')
    return profile['fixed_indices']


def assert_slab(atoms):
    prepared = _assert_saved_surface_profile(atoms)
    if prepared is not None:
        return prepared
    ads = atoms.info.get('adsorbate_indices', [])
    slab = [i for i in range(len(atoms)) if i not in ads]
    fixed = sorted(set(range(len(atoms))) - set(free_indices(atoms)))
    if len(slab) != 36 or any((atoms[i].symbol != 'Ni' for i in slab)) or len(fixed) != 18:
        raise ValueError('Requires Ni 3x3x4 with the bottom 18 Ni atoms fixed')
    bottom = sorted(slab, key=lambda i: atoms.positions[i, 2])[:18]
    if fixed != sorted(bottom):
        raise ValueError('Bottom 2-layer constraint mismatch')
    from chavis.judgment.surface_context import selected_surface_element
    if selected_surface_element(from_ase(atoms)) != 'Ni':
        raise ValueError('Unprofiled Ni surface is not the verified legacy Ni(111) terrace')
    return fixed


def ads_edges(atoms, *, criterion_profile=V2_PROFILE):
    ads = atoms.info['adsorbate_indices']
    edges = []
    for n, i in enumerate(ads):
        for j in ads[n + 1:]:
            if bond_intact(
                atoms.get_distance(i, j, mic=True),
                atoms[i].symbol,
                atoms[j].symbol,
                nonreference_scale=_BOND_SCALE,
                criterion_profile=criterion_profile,
                inclusive=False
            ):
                edges.append((int(i), int(j)))
    return edges


def mapped_rms(a, b, indices):
    v, _ = find_mic(a.positions[indices] - b.positions[indices], a.cell, a.pbc)
    return float(np.sqrt(np.mean(np.sum(v * v, axis=1))))


def initial_frames(
    initial,
    final,
    total,
    seed=None,
    *,
    policy: IDPPPolicy=NORMAL_REACTION_MIC_IDPP
):

    def segment(a, b, count):
        frames, _ = prepare_idpp(from_ase(a), from_ase(b), count, policy=policy)
        return frames
    if seed is None:
        frames = segment(initial, final, total)
    else:
        left = total // 2 + 1
        frames = segment(initial, seed, left)[:-1] + segment(seed, final, total - left + 1)
    fixed = sorted(set(range(len(initial))) - set(free_indices(initial)))
    for frame in frames:
        for i in fixed:
            frame['positions'][i] = initial.positions[i].tolist()
    return frames
