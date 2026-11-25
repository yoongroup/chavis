from __future__ import annotations
import os
import time
from . import config as C
_UNIT = None
_CALCS: dict = {}
_LOAD_SECONDS = None


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec('fairchem') is not None


def get_calculator(task: str):
    global _UNIT, _LOAD_SECONDS
    if task in _CALCS:
        return _CALCS[task]
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    if _UNIT is None:
        t0 = time.time()
        settings = InferenceSettings(
            compile=C.UMA_COMPILE,
            merge_mole=C.UMA_COMPILE,
            tf32=False
        )
        _UNIT = pretrained_mlip.get_predict_unit(
            C.UMA_MODEL,
            inference_settings=settings,
            device=C.UMA_DEVICE
        )
        _LOAD_SECONDS = time.time() - t0
    _CALCS[task] = FAIRChemCalculator(_UNIT, task_name=task)
    return _CALCS[task]


def load_seconds():
    return _LOAD_SECONDS


def relax_gas(frag):
    import numpy as np
    from ase import Atoms
    from ase.optimize import BFGS
    w = Atoms(frag.raw_symbols, positions=frag.raw_positions)
    w.pbc = False
    w.info.update(charge=frag.charge, spin=frag.spin)
    w.calc = get_calculator(C.GAS_TASK)
    e0 = float(w.get_potential_energy())
    before = w.get_positions().copy()
    opt = BFGS(w, logfile=None)
    if callable(globals().get('CHECK_RUN')):
        opt.attach(CHECK_RUN, interval=1)
    conv = bool(opt.run(fmax=C.GAS_FMAX, steps=C.GAS_STEPS))
    e1 = float(w.get_potential_energy())
    moved = float(np.linalg.norm(w.get_positions() - before, axis=1).max())
    n = int(opt.get_number_of_steps())
    frag.raw_positions = [list(p) for p in w.get_positions()]
    frag.positions = [list(p) for p in w.get_positions()]
    frag.log(
        3,
        'UMA gas-phase optimization',
        f"{C.GAS_TASK} fmax={C.GAS_FMAX} ({n} steps) {('converged' if conv else 'not converged')}, E {e0:.3f} → {e1:.3f} eV, maximum displacement {moved:.3f} Å",
        'ok' if conv else 'warn'
    )
    w.calc = None
    return {
        'atoms': w,
        'e_before': e0,
        'e_after': e1,
        'converged': conv,
        'steps': n,
        'moved_max': moved
    }


def relax_surface(atoms, n_slab: int, binder: int=0):
    import numpy as np
    from ase.optimize import BFGS
    w = atoms.copy()
    if not w.constraints:
        raise ValueError('Missing bottom-two-layer constraint')
    w.calc = get_calculator(C.SURF_TASK)
    e0 = float(w.get_potential_energy())
    before = w.get_positions().copy()
    opt = BFGS(w, logfile=None)
    if callable(globals().get('CHECK_RUN')):
        opt.attach(CHECK_RUN, interval=1)
    conv = bool(opt.run(fmax=C.SURF_FMAX, steps=C.SURF_STEPS))
    e1 = float(w.get_potential_energy())
    disp = np.linalg.norm(w.get_positions()[n_slab:] - before[n_slab:], axis=1)
    z = w.get_positions()
    top = float(max((z[i][2] for i in range(n_slab))))
    height = float(z[n_slab + binder][2] - top)
    w.calc = None
    info = {
        'e_before': e0,
        'e_after': e1,
        'converged': conv,
        'steps': int(opt.get_number_of_steps()),
        'moved_max': float(disp.max()),
        'height': height
    }
    return (w, info)


def classify(info) -> str:
    if not info.get('converged'):
        return 'unconverged'
    if info['height'] > C.DESORBED_A:
        return 'desorbed'
    if info['moved_max'] > C.MIGRATED_A:
        return 'migrated'
    return 'kept'


def rank_sites(energies: dict) -> list:
    if not energies:
        return []
    lo = min(energies.values())
    ordered = sorted(energies.items(), key=lambda kv: kv[1])
    groups, cur = ([], [ordered[0]])
    for name, e in ordered[1:]:
        if e - cur[0][1] <= C.DEGENERATE_EV:
            cur.append((name, e))
        else:
            groups.append(cur)
            cur = [(name, e)]
    groups.append(cur)
    return [
        (i + 1, [n for n, _ in g], round(g[0][1] - lo, 4)) for i,
        g in enumerate(groups)
    ]
