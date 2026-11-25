from __future__ import annotations
import itertools
import math
from .geometry import RADII, add, cross, dot, mul, norm, sub, unit


class GeometryError(ValueError):
    """"""


def dual(s):
    a, b, _ = s['cell']
    if norm(cross(a, b)) < 1e-08:
        raise GeometryError('A periodic surface cell is required. Use extxyz or a QE input; plain XYZ has no cell.')
    n = unit(cross(a, b))
    det = dot(a, cross(b, n))
    return (mul(cross(b, n), 1 / det), mul(cross(n, a), 1 / det), n)


def translations(v, s, radius):
    g0, g1, _ = dual(s)
    ranges = []
    for j, g in enumerate((g0, g1)):
        if not s['pbc'][j]:
            ranges.append([0])
            continue
        f = dot(v, g)
        bound = radius * norm(g) + 1e-08
        ranges.append(range(math.ceil(f - bound), math.floor(f + bound) + 1))
    if len(ranges[0]) * len(ranges[1]) > 20000:
        raise GeometryError('Cell/search radius gives too many periodic images. Reduce radius or use a better-conditioned supercell.')
    for i, j in itertools.product(*ranges):
        yield ((i, j), add(mul(s['cell'][0], i), mul(s['cell'][1], j)))


def minimum_vector(v, s):
    g0, g1, _ = dual(s)
    ns = [round(dot(v, g)) if s['pbc'][i] else 0 for i, g in enumerate((g0, g1))]
    best = sub(v, add(mul(s['cell'][0], ns[0]), mul(s['cell'][1], ns[1])))
    best2 = dot(best, best)
    for _, t in translations(v, s, math.sqrt(best2)):
        q = sub(v, t)
        q2 = dot(q, q)
        if q2 < best2 - 1e-12:
            best, best2 = (q, q2)
    return best


def collisions(s, moving, scale):
    movable = set(moving)
    pairs = []
    for i in moving:
        for j in range(len(s['symbols'])):
            if j in movable:
                continue
            d = norm(minimum_vector(sub(s['positions'][i], s['positions'][j]), s))
            cutoff = scale * (RADII.get(s['symbols'][i], 1.2) + RADII.get(s['symbols'][j], 1.2))
            if d < cutoff:
                pairs.append([i, j, round(d, 4)])
    for i in moving:
        for j in moving:
            v = sub(s['positions'][i], s['positions'][j])
            cutoff = scale * (RADII.get(s['symbols'][i], 1.2) + RADII.get(s['symbols'][j], 1.2))
            for ij, t in translations(v, s, cutoff):
                if ij == (0, 0):
                    continue
                d = norm(sub(v, t))
                if d < cutoff:
                    pairs.append([i, j, round(d, 4), 'periodic_image'])
    return pairs
