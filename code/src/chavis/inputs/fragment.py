from __future__ import annotations
import math
from dataclasses import dataclass, field
from . import prep_config as C


@dataclass
class Fragment:
    name: str
    smiles: str
    mode: str
    symbols: list
    positions: list
    binder: int
    valence: int
    charge: int
    spin: int
    spin_reason: str
    raw_symbols: list = field(default_factory=list)
    raw_positions: list = field(default_factory=list)
    holders: list = field(default_factory=list)
    steps: list = field(default_factory=list)

    def log(self, n, what, result, status='ok'):
        self.steps.append((str(n), what, result, status))


def build_fragment(name: str, smiles: str, mode: str) -> Fragment:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'{name}: SMILES parsing failed ({smiles})')
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() > 100:
        raise ValueError('Parent preparation limit: 100 atoms including placeholders')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('Prepare one connected molecular component at a time')
    holders = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == C.PLACEHOLDER]
    if mode == 'Cl':
        if not holders:
            raise ValueError(f'{name}: Cl substitution mode requires {C.PLACEHOLDER} which is absent')
        binders = {n.GetIdx() for h in holders for n in mol.GetAtomWithIdx(h).GetNeighbors()}
        if len(binders) != 1:
            raise ValueError(f'{name}: {C.PLACEHOLDER} are attached to distinct atoms: {len(binders)}. They must attach to one atom')
        binder = binders.pop()
    elif mode.startswith('free:'):
        if holders:
            raise ValueError('Cl-containing molecules require explicit placeholder handling; free mode does not remove Cl.')
        want = mode.split(':', 1)[1].strip()
        cand = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == want]
        if not cand:
            raise ValueError(f'{name}: free-mode binding element {want} is absent from the molecule')
        binder = cand[0]
    else:
        raise ValueError('mode must be Cl or free:Element')
    params = AllChem.ETKDGv3()
    ok, attempt = (-1, 0)
    for attempt in range(1, C.EMBED_TRIES + 1):
        params.randomSeed = C.EMBED_SEED + attempt - 1
        ok = AllChem.EmbedMolecule(mol, params)
        if ok == 0:
            break
    if ok != 0:
        raise ValueError(f'{name}: 3D embedding failed ({C.EMBED_TRIES} attempts)')
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = [list(p) for p in mol.GetConformer().GetPositions()]
    charge = Chem.GetFormalCharge(mol)
    n_rad = Descriptors.NumRadicalElectrons(mol)
    spin, reason = (n_rad + 1, f'RDKit radical {n_rad} items')
    if mol.GetNumAtoms() == 1 and mol.GetAtomWithIdx(0).GetSymbol() in C.SPIN_OVERRIDE:
        spin = C.SPIN_OVERRIDE[mol.GetAtomWithIdx(0).GetSymbol()]
        reason = f'override (RDKit reports {n_rad + 1})'
    f = Fragment(
        name=name,
        smiles=smiles,
        mode=mode,
        symbols=list(sym),
        positions=[list(p) for p in pos],
        binder=binder,
        valence=len(holders) if mode == 'Cl' else 0,
        charge=charge,
        spin=spin,
        spin_reason=reason,
        raw_symbols=list(sym),
        raw_positions=[list(p) for p in pos],
        holders=list(holders)
    )
    f.log(
        1,
        'SMILES parsing',
        f'Atoms: {len(sym)}, ETKDGv3 seed={params.randomSeed:#x} (attempt {attempt})'
    )
    if mode == 'Cl':
        kind = {1: 'top', 2: 'bridge'}.get(len(holders), 'hollow')
        f.log(
            2,
            'Placeholder atoms',
            f'{C.PLACEHOLDER} {len(holders)} attached to {sym[binder]}{binder}; open valence {len(holders)}, {kind} tendency'
        )
    else:
        f.log(
            2,
            'Placeholder atoms',
            f'none (free mode). Binding atom set to {sym[binder]}{binder} selected',
            'warn'
        )
    f.log(2, 'charge/spin', f'charge={charge} spin={spin} [{reason}]')
    return f


def finalize_direction(f: Fragment) -> Fragment:
    import numpy as np
    pos = np.asarray(f.positions, float)
    sym = list(f.symbols)
    binder = f.binder
    if f.mode == 'Cl':
        vecs = []
        for h in f.holders:
            v = pos[h] - pos[binder]
            n = float(np.linalg.norm(v))
            if n < 1e-06:
                raise ValueError(f'{f.name}: placeholder and binding atom overlap')
            vecs.append(v / n)
        axis = np.sum(vecs, axis=0)
        if np.linalg.norm(axis) < 1e-06:
            raise ValueError(f'{f.name}: {C.PLACEHOLDER} directions cancel by symmetry')
        axis = axis / np.linalg.norm(axis)
        spread = [math.degrees(math.acos(float(np.clip(v @ axis, -1, 1)))) for v in vecs]
        f.log(
            4,
            'Binding direction',
            f'({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}), each placeholder from the mean: ' + ', '.join((f'{s:.1f}°' for s in spread))
        )
    else:
        axis = np.array([0.0, 0.0, -1.0])
        f.log(
            4,
            'Binding direction',
            'Free mode preserves embedded orientation and translates the binding atom to the origin; lowest-atom status is recorded separately.',
            'warn'
        )
    if f.mode == 'Cl':
        keep = [i for i in range(len(sym)) if i not in set(f.holders)]
        binder = keep.index(binder)
        pos = pos[keep]
        sym = [sym[i] for i in keep]
        f.log(
            4,
            'Placeholder removal',
            f'{C.PLACEHOLDER} {len(f.holders)} removed, atoms remaining: {len(sym)} remaining'
        )
    rot, deg = _rotation(axis, np.array([0.0, 0.0, -1.0]))
    pos = (pos - pos[binder]) @ rot.T
    low = int(np.argmin(pos[:, 2]))
    ok_low = low == binder
    note = '' if ok_low else f' ← {sym[low]}{low} is {pos[binder, 2] - pos[low, 2]:.3f} Angstrom lower and contacts first on dropping'
    f.log(
        4,
        'Alignment',
        f'Rotation {deg:.1f} degrees, lowest atom {sym[low]}{low}{note}',
        'ok' if ok_low else 'warn'
    )
    f.symbols, f.positions, f.binder = (sym, [list(p) for p in pos], binder)
    return f


def _rotation(a, b):
    import numpy as np
    axis = np.cross(a, b)
    s, c = (float(np.linalg.norm(axis)), float(a @ b))
    if s < 1e-09:
        return (np.eye(3), 0.0) if c > 0 else (np.diag([1.0, -1.0, -1.0]), 180.0)
    axis = axis / s
    k = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    th = math.atan2(s, c)
    return (np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k, math.degrees(th))


def parse_species(path) -> list:
    import re
    from pathlib import Path
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        p = re.split('\\s{2,}|\\t', line)
        if len(p) < 3:
            continue
        rows.append((
            p[0].strip(),
            p[1].strip(),
            p[2].strip(),
            p[3].strip() if len(p) > 3 else ''
        ))
    return rows
