import numpy as np
from ase.geometry import find_mic
from chavis.runtime.json_io import append


def record_neb(run, optimizer, band, images, stage):
    if not hasattr(band, 'real_forces'):
        return
    links = []
    for left, right in zip(images, images[1:]):
        direct = right.positions - left.positions
        mic, _ = find_mic(direct, left.cell, left.pbc)
        shift = np.rint((mic - direct) @ np.linalg.inv(left.cell.array)).astype(int)
        links.append({
            'ads_MIC_RMS_A': float(np.sqrt(np.mean(np.sum(mic[36:] ** 2, axis=1)))),
            'H37_MIC_vector_A': mic[37].tolist(),
            'H37_cell_shift': shift[37].tolist(),
            'C36_MIC_vector_A': mic[36].tolist()
        })
    state = {}
    for key in (
        'dt',
        'dtmax',
        'a',
        'astart',
        'fa',
        'finc',
        'fdec',
        'Nsteps',
        'Nmin',
        'maxstep',
        'downhill_check',
        'v'
    ):
        value = getattr(optimizer, key, None)
        state[key] = value.tolist() if isinstance(value, np.ndarray) else value
    bonds = []
    for atoms in images:
        vectors, _ = find_mic(
            atoms.positions[37:41] - atoms.positions[36],
            atoms.cell,
            atoms.pbc
        )
        bonds.append(np.linalg.norm(vectors, axis=1).tolist())
    append(
        run.d / 'neb_diagnostics.jsonl',
        {
            'stage': stage,
            'step': optimizer.nsteps,
            'FIRE_state': state,
            'links': links,
            'C36_H37_to_H40_distances_A': bonds,
            'raw_force_source': stage + '.traj; full per-image forces retained by ASE trajectory',
            'diagnostic_only': True,
            'model_calls': 0
        }
    )
