from __future__ import annotations
import copy, io, itertools, json, math, re, shlex
RADII = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'Ni': 1.24,
    'Cu': 1.32,
    'Pd': 1.39,
    'Pt': 1.36,
    'Rh': 1.42,
    'Fe': 1.32,
    'Co': 1.26,
    'S': 1.05,
    'P': 1.07
}


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def mul(a, s):
    return [v * s for v in a]


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)))


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n < 1e-12:
        raise ValueError('Cannot normalize a zero-length direction. Choose a different reference bond.')
    return mul(a, 1 / n)


def basis(s):
    cell = s.get('cell', [[0, 0, 0]] * 3)
    if norm(cell[0]) > 1e-10 and norm(cross(cell[0], cell[1])) > 1e-10:
        x = unit(cell[0])
        z = unit(cross(cell[0], cell[1]))
        y = cross(z, x)
    else:
        x, y, z = ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])
    return (x, y, z)


def rotate(v, axis, angle):
    u = unit(axis)
    c = math.cos(angle)
    sn = math.sin(angle)
    return add(add(mul(v, c), mul(cross(u, v), sn)), mul(u, dot(u, v) * (1 - c)))


def parse_indices(value, n):
    if isinstance(value, list):
        out = [int(x) for x in value]
    else:
        out = []
        for token in str(value or '').replace(';', ',').split(','):
            token = token.strip()
            if not token:
                continue
            if ':' in token:
                start, stop = map(int, token.split(':'))
                out.extend(range(start, stop))
            else:
                out.append(int(token))
    out = list(dict.fromkeys(out))
    if any((i < 0 or i >= n for i in out)):
        raise ValueError(f'Atom index outside 0..{n - 1}. All indices are zero based.')
    return out


def validate(s):
    if not isinstance(s, dict):
        raise ValueError('A structure must be a dictionary.')
    sy = s.get('symbols', [])
    pos = s.get('positions', [])
    if not sy or len(sy) != len(pos):
        raise ValueError('symbols and positions must have the same nonzero length.')
    if len(sy) > 10000:
        raise ValueError('Preview limit: 10,000 atoms in one structure.')
    if any((not isinstance(e, str) or not re.fullmatch('[A-Z][a-z]?', e) for e in sy)):
        raise ValueError('Invalid element symbol.')
    p = [[float(v) for v in row] for row in pos]
    if any((len(row) != 3 or any((not math.isfinite(v) for v in row)) for row in p)):
        raise ValueError('Positions must be finite N x 3 numbers.')
    cell = [[float(v) for v in row] for row in s.get('cell', [[0, 0, 0]] * 3)]
    if len(cell) != 3 or any((len(row) != 3 or any((not math.isfinite(v) for v in row)) for row in cell)):
        raise ValueError('Invalid 3 x 3 cell.')
    pbc = s.get('pbc', [False] * 3)
    if len(pbc) != 3:
        raise ValueError('pbc must have three entries.')
    ids = [int(x) for x in s.get('atom_ids', range(len(sy)))]
    if len(ids) != len(sy) or len(set(ids)) != len(ids):
        raise ValueError('atom_ids must be unique and have length N.')
    ans = copy.deepcopy(s)
    ans.update(
        symbols=sy,
        positions=p,
        cell=cell,
        pbc=[bool(x) for x in pbc],
        atom_ids=ids
    )
    return ans


def read_xyz(text):
    lines = text.splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].strip())
        except ValueError:
            raise ValueError(f'Line {i + 1}: expected an XYZ atom count.')
        if n <= 0 or n > 10000 or i + n + 2 > len(lines):
            raise ValueError('Invalid atom count or truncated XYZ frame.')
        comment = lines[i + 1]
        fields = {}
        for token in shlex.split(comment):
            if '=' in token:
                k, v = token.split('=', 1)
                fields[k] = v
        desc = fields.get('Properties', 'species:S:1:pos:R:3').split(':')
        specs = []
        offset = 0
        if len(desc) % 3:
            raise ValueError('Malformed extended XYZ Properties field.')
        for j in range(0, len(desc), 3):
            count = int(desc[j + 2])
            specs.append((desc[j], offset, count))
            offset += count
        sy = []
        pos = []
        ids = []
        fixed = []
        magmoms = []
        for k in range(n):
            cols = lines[i + 2 + k].split()
            props = {name: cols[off:off + c] for name, off, c in specs}
            sym = props.get('species', props.get('element', []))
            xyz = props.get('pos', props.get('positions', []))
            if len(sym) != 1 or len(xyz) != 3:
                raise ValueError('XYZ requires species and pos columns.')
            sy.append(sym[0])
            pos.append([float(x) for x in xyz])
            ids.append(int(props.get('rw_id', props.get('id', [k]))[0]))
            if props.get('rw_fixed', ['F'])[0].lower() in ('t', 'true', '1') or props.get('move_mask', ['T'])[0].lower() in ('f', 'false', '0'):
                fixed.append(k)
            if 'initial_magmoms' in props:
                magmoms.append(float(props['initial_magmoms'][0]))
        lattice = [float(x) for x in fields.get('Lattice', '0 0 0 0 0 0 0 0 0').split()]
        if len(lattice) != 9:
            raise ValueError('Lattice must contain 9 numbers.')
        pbc = [x.lower() in ('t', 'true', '1') for x in fields.get('pbc', 'F F F').split()]
        if len(pbc) == 1:
            pbc *= 3
        s = {
            'symbols': sy,
            'positions': pos,
            'atom_ids': ids,
            'cell': [lattice[j:j + 3] for j in (0, 3, 6)],
            'pbc': pbc,
            'meta': {
                'source_comment': comment,
                'atom_id_source': 'file' if any((k in fields.get('Properties', '') for k in ['rw_id:', ':id:'])) else 'row-order'
            },
            'fixed_indices': fixed
        }
        if len(magmoms) == n:
            s['initial_magmoms'] = magmoms
        for key in ['energy', 'Energy', 'E']:
            try:
                if key in fields:
                    s['energy'] = float(fields[key])
            except ValueError:
                pass
        if 'rw_meta' in fields:
            try:
                s['meta'].update(json.loads(fields['rw_meta']))
            except (ValueError, TypeError):
                pass
        if 'adsorbate_indices' in s['meta']:
            s['adsorbate_indices'] = s['meta']['adsorbate_indices']
        frames.append(validate(s))
        i += n + 2
        if len(frames) > 2000:
            raise ValueError('Import limit: 2,000 frames per file. Split a large trajectory first.')
    if not frames:
        raise ValueError('No XYZ frames found.')
    return frames


def write_xyz(s):
    s = validate(s)
    meta = dict(s.get('meta', {}))
    meta['adsorbate_indices'] = s.get('adsorbate_indices', [])
    if s.get('meta', {}).get('unsupported_constraints'):
        raise ValueError('Unsupported constraints are recorded. Review them before exporting; this prototype only exports fixed-atom constraints.')
    header = 'Lattice="' + ' '.join((f'{x:.12g}' for row in s['cell'] for x in row)) + '" Properties=species:S:1:pos:R:3:rw_id:I:1:rw_fixed:L:1:move_mask:L:1' + (':initial_magmoms:R:1' if s.get('initial_magmoms') else '') + ' pbc="' + ' '.join(('T' if x else 'F' for x in s['pbc'])) + '"'
    header += ' rw_meta=' + shlex.quote(json.dumps(
        meta,
        ensure_ascii=False,
        separators=(',', ':')
    ))
    if 'energy' in s:
        header += f" energy={float(s['energy']):.14g}"
    lines = [str(len(s['symbols'])), header]
    fixed = set(s.get('fixed_indices', []))
    for i, (sym, p, aid) in enumerate(zip(s['symbols'], s['positions'], s['atom_ids'])):
        lines.append(f'{sym} ' + ' '.join((f'{v:.12f}' for v in p)) + f' {aid} ' + ('T' if i in fixed else 'F') + ' ' + ('F' if i in fixed else 'T') + (f" {s['initial_magmoms'][i]:.12g}" if s.get('initial_magmoms') else ''))
    return '\n'.join(lines) + '\n'


def to_ase(s):
    from ase import Atoms
    from ase.constraints import FixAtoms
    import numpy as np
    s = validate(s)
    if s.get('meta', {}).get('unsupported_constraints'):
        raise ValueError('Input has unsupported constraints. Review before conversion to ASE.')
    a = Atoms(
        symbols=s['symbols'],
        positions=s['positions'],
        cell=s['cell'],
        pbc=s['pbc']
    )
    a.set_array('rw_id', np.array(s['atom_ids'], dtype=int))
    if s.get('fixed_indices'):
        a.set_constraint(FixAtoms(indices=s['fixed_indices']))
    if s.get('initial_magmoms'):
        a.set_initial_magnetic_moments(s['initial_magmoms'])
    a.info.update(s.get('meta', {}))
    a.info['adsorbate_indices'] = s.get('adsorbate_indices', [])
    return a


def from_ase(a):
    ids = a.arrays.get('rw_id')
    fixed = []
    unsupported = []
    from ase.constraints import FixAtoms
    for con in a.constraints:
        if isinstance(con, FixAtoms):
            fixed.extend(con.get_indices().tolist())
        else:
            unsupported.append(type(con).__name__)
    s = {
        'symbols': a.get_chemical_symbols(),
        'positions': a.positions.tolist(),
        'cell': a.cell.array.tolist(),
        'pbc': a.pbc.tolist(),
        'atom_ids': ids.tolist() if ids is not None else list(range(len(a))),
        'fixed_indices': fixed,
        'meta': {'atom_id_source': 'file' if ids is not None else 'row-order'}
    }
    for key, value in a.info.items():
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            continue
        s['meta'][key] = copy.deepcopy(value)
    if isinstance(a.info.get('adsorbate_indices'), list):
        s['adsorbate_indices'] = a.info['adsorbate_indices'][:]
    if unsupported:
        s['meta']['unsupported_constraints'] = unsupported
    if 'initial_magmoms' in a.arrays:
        s['initial_magmoms'] = a.get_initial_magnetic_moments().tolist()
    if a.calc is not None:
        results = a.calc.results
        if 'energy' in results:
            s['energy'] = float(results['energy'])
    return validate(s)


def read_any(path, format_hint=None):
    from pathlib import Path
    path = Path(path)
    if path.suffix.lower() in ('.xyz', '.extxyz'):
        return read_xyz(path.read_text(encoding='utf-8-sig'))
    try:
        from ase.io import iread
    except ImportError:
        raise ValueError('This format needs ASE. Run: python3 -m pip install ase (inside this app environment). XYZ/extxyz works without ASE.')
    fmt = format_hint or None
    if not fmt:
        ext = path.suffix.lower()
        if ext in ('.out', '.pwo'):
            fmt = 'espresso-out'
        elif ext in ('.in', '.pwi'):
            fmt = 'espresso-in'
    kwargs = {'results_required': False} if fmt == 'espresso-out' else {}
    frames = []
    for a in iread(str(path), index=':', format=fmt, **kwargs):
        frames.append(from_ase(a))
        if len(frames) > 2000:
            raise ValueError('Import limit is 2,000 frames per file.')
    if not frames:
        raise ValueError('No structures could be read. Check the file format.')
    return frames


def compatible(a, b):
    if a['symbols'] != b['symbols']:
        raise ValueError('Atom count or element order differs. Provide explicitly atom-mapped endpoints.')
    if a.get('atom_ids') != b.get('atom_ids'):
        raise ValueError('Atom IDs differ. Explicit remapping is required; no automatic permutation is performed.')
    if a.get('pbc') != b.get('pbc'):
        raise ValueError('PBC flags differ between the endpoints.')
    if any((
        abs(x - y) > 1e-06 for ra,
        rb in zip(a['cell'], b['cell']) for x,
        y in zip(ra, rb)
    )):
        raise ValueError('Cell differs. This prototype does not interpolate changing cells.')


def interpolate(a, b, n=8):
    compatible(a, b)
    if not 3 <= n <= 40:
        raise ValueError('Use 3 to 40 total images, including endpoints.')
    images = []
    for k in range(n):
        s = copy.deepcopy(a)
        t = k / (n - 1)
        s['positions'] = [
            add(mul(x, 1 - t), mul(y, t)) for x,
            y in zip(a['positions'], b['positions'])
        ]
        s.pop('energy', None)
        s['meta'] = {
            **s.get('meta', {}),
            'path_type': 'Cartesian linear guess — NOT a MEP/TS',
            'image': k
        }
        images.append(s)
    return images


def example_mothers():
    spacing = 3.52 / math.sqrt(2)
    x = [spacing, 0, 0]
    y = [spacing / 2, spacing * math.sqrt(3) / 2, 0]
    dz = 3.52 / math.sqrt(3)
    positions = []
    for layer in range(4):
        shift = mul(add(x, y), layer % 3 / 3)
        for i in range(3):
            for j in range(3):
                positions.append(add(
                    add(mul(x, i), mul(y, j)),
                    add(shift, [0, 0, 5 + layer * dz])
                ))
    cell = [mul(x, 3), mul(y, 3), [0, 0, 22.0]]
    top = max((p[2] for p in positions))
    anchor = add(mul(add(cell[0], cell[1]), 0.43), [0, 0, top + 1.8])
    mothers = []
    templates = {
        'CH3': (
            ['C', 'H', 'H', 'H'],
            [[0, 0, 0], [1.02, 0, 0.37], [-0.51, 0.883, 0.37], [-0.51, -0.883, 0.37]]
        ),
        'OH': (['O', 'H'], [[0, 0, 0], [0.82, 0, 0.52]]),
        'CO2': (['C', 'O', 'O'], [[0, 0, 0], [1.05, 0, 0.35], [-1.05, 0, 0.35]])
    }
    for species, (sy, off) in templates.items():
        for angle in [0, 60, 120] if species == 'CH3' else [0]:
            points = positions + [add(
                anchor,
                rotate(v, [0, 0, 1], math.radians(angle))
            ) for v in off]
            s = validate({
                'symbols': ['Ni'] * 36 + sy,
                'positions': points,
                'cell': cell,
                'pbc': [True, True, False],
                'fixed_indices': list(range(9)),
                'adsorbate_indices': list(range(36, 36 + len(sy))),
                'meta': {
                    'demo': True,
                    'note': 'Synthetic unrelaxed geometry. No energy, no calculated minimum.',
                    'species': species,
                    'azimuth_deg': angle
                }
            })
            mothers.append({
                'name': f'{species}/mother_{angle:03d}.extxyz',
                'structure': s
            })
    return mothers
