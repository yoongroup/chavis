from __future__ import annotations
import json
import os
from pathlib import Path


def comparison_context(s, charge=0):
    fixed = s['fixed_indices']
    return {
        'composition': s['symbols'],
        'cell': s['cell'],
        'pbc': s['pbc'],
        'fixed_indices': fixed,
        'fixed_positions': [s['positions'][i] for i in fixed],
        'atom_ids': s['atom_ids'],
        'model': 'uma-s-1p2p1',
        'task': 'oc20',
        'charge': charge,
        'initial_magmoms': s.get('initial_magmoms'),
        'constraints': 'FixAtoms'
    }


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False
    ))
    temporary.replace(path)
