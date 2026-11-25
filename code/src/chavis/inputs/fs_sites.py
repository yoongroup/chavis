from __future__ import annotations
import numpy as np
from ase.constraints import FixAtoms
from chavis.search.geometry import from_ase
from chavis.search.collision import collisions, minimum_vector
from chavis.search.ts_helpers import assert_slab
from .contact_placement import contact_target
from .library_sites import enumerate_sites
from .seed_geometry import surface_frame


def direct_fs_seeds(atoms, moving, anchor, settings, preferred_site):
    structure = from_ase(atoms)
    ads = structure['adsorbate_indices']
    normal, plane, _ = surface_frame(atoms)
    sites, backend = enumerate_sites(
        structure,
        ads,
        {'site_backend': 'existing_terrace'}
    )
    rows = []
    for site in sites:
        vector = np.asarray(minimum_vector(
            (np.asarray(site['position']) - atoms.positions[anchor]).tolist(),
            structure
        ))
        lateral = float(np.linalg.norm(vector - vector @ normal * normal))
        if not settings['rmin'] <= lateral <= settings['rmax']:
            continue
        target, contact = contact_target(
            structure,
            moving,
            ads,
            site['position'],
            normal.tolist(),
            1.0
        )
        trial = atoms.copy()
        trial.positions[moving] += np.asarray(target) - trial.positions[moving[0]]
        trial.set_constraint(FixAtoms(indices=assert_slab(atoms)))
        saved = from_ase(trial)
        collision_pairs = collisions(saved, moving, 0.7)
        rows.append({
            'structure': saved,
            'site': site,
            'lateral_distance_A': lateral,
            'contact': contact,
            'collision_pairs': collision_pairs,
            'accepted': not collision_pairs
        })
    wanted = 'top' if preferred_site == 'ontop' else preferred_site
    rows.sort(key=lambda row: (
        row['site']['kind'] != wanted,
        row['lateral_distance_A'],
        tuple(row['site']['position'])
    ))
    return (
        rows,
        {
            'backend': backend,
            'reference_plane_A': plane.tolist(),
            'normal': normal.tolist(),
            'angle_scope': 'direct coadsorbed FS; no molecular cleavage azimuth is assumed',
            'contact_scale': 1.0,
            'collision_scale': 0.7,
            'rmin': settings['rmin'],
            'rmax': settings['rmax']
        }
    )
