from __future__ import annotations
import itertools
import numpy as np
DOWN = np.array([0.0, 0.0, -1.0])
_EPS = 1e-06


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < _EPS else np.asarray(v, float) / n


def _perpendicular(v):
    v = _unit(v)
    e = np.eye(3)[int(np.argmin(np.abs(v)))]
    return _unit(e - e @ v * v)


def rotation_to(a, b):
    a, b = (_unit(a), _unit(b))
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(np.clip(a @ b, -1, 1))
    if s < 1e-09:
        if c > 0:
            return np.eye(3)
        p = _perpendicular(a)
        return 2 * np.outer(p, p) - np.eye(3)
    axis /= s
    k = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + s * k + (1 - c) * k @ k


def rotation_z(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def _dimension(p):
    if len(p) == 1:
        return (0, None)
    q = p - p.mean(axis=0)
    _, s, vt = np.linalg.svd(q)
    rank = int(np.sum(s > 0.05))
    return (rank, vt)


def hull_features(positions):
    p = np.asarray(positions, float)
    centre = p.mean(axis=0)
    dim, vt = _dimension(p)
    out = []
    if dim == 0:
        return [('vertex', [0], DOWN.copy())]
    if dim == 1:
        axis = vt[0]
        t = (p - centre) @ axis
        lo, hi = (int(np.argmin(t)), int(np.argmax(t)))
        out.append(('vertex', [lo], _unit(p[lo] - centre)))
        out.append(('vertex', [hi], _unit(p[hi] - centre)))
        out.append(('edge', sorted([lo, hi]), _perpendicular(axis)))
        return out
    if dim == 2:
        normal = vt[2]
        u, w = (vt[0], vt[1])
        xy = np.c_[(p - centre) @ u, (p - centre) @ w]
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xy)
        verts = list(hull.vertices)
        for i in verts:
            d = p[i] - centre
            out.append(('vertex', [int(i)], _unit(d - d @ normal * normal)))
        for a, b in hull.simplices:
            mid = (p[a] + p[b]) / 2 - centre
            d = mid - mid @ normal * normal
            edge = p[b] - p[a]
            d = d - d @ _unit(edge) * _unit(edge)
            out.append(('edge', sorted([int(a), int(b)]), _unit(d)))
        members = sorted((int(i) for i in range(len(p))))
        out.append(('face', members, _unit(normal)))
        out.append(('face', members, _unit(-normal)))
        return [f for f in out if f[2] is not None]
    from scipy.spatial import ConvexHull
    hull = ConvexHull(p)
    for i in hull.vertices:
        d = _unit(p[i] - centre)
        if d is not None:
            out.append(('vertex', [int(i)], d))
    edges = set()
    for simplex in hull.simplices:
        for a, b in itertools.combinations(sorted((int(x) for x in simplex)), 2):
            edges.add((a, b))
    normals = [np.asarray(eq[:3]) for eq in hull.equations]
    facet_of = {}
    for k, simplex in enumerate(hull.simplices):
        for a, b in itertools.combinations(sorted((int(x) for x in simplex)), 2):
            facet_of.setdefault((a, b), []).append(k)
    for a, b in sorted(edges):
        ks = facet_of[a, b]
        if len(ks) == 2 and abs(normals[ks[0]] @ normals[ks[1]]) > 1 - 1e-06:
            continue
        n = _unit(sum((normals[k] for k in ks)))
        edge = _unit(p[b] - p[a])
        if n is None:
            continue
        n = _unit(n - n @ edge * edge)
        if n is not None:
            out.append(('edge', [a, b], n))
    faces = {}
    for k, eq in enumerate(hull.equations):
        key = tuple(np.round(eq[:3], 4))
        faces.setdefault(key, set()).update((int(x) for x in hull.simplices[k]))
    for key, members in faces.items():
        out.append(('face', sorted(members), _unit(np.asarray(key))))
    return out


def reference_point(positions, members):
    return np.asarray(positions, float)[list(members)].mean(axis=0)


def generate(positions, *, placeholder_binder=None, azimuth_step_deg=45.0):
    p = np.asarray(positions, float)
    feats = hull_features(p)
    if placeholder_binder is not None:
        feats = [('placeholder_down', [int(placeholder_binder)], DOWN.copy())] + feats
    steps = max(1, int(round(360.0 / azimuth_step_deg)))
    out = []
    for index, (kind, members, down) in enumerate(feats):
        r = rotation_to(down, DOWN)
        q = (p - p.mean(axis=0)) @ r.T
        ref = reference_point(q, members)
        q = q - ref
        for k in range(steps):
            az = k * azimuth_step_deg
            out.append({
                'feature': kind,
                'feature_index': index,
                'members': list(members),
                'azimuth_deg': az,
                'positions': q @ rotation_z(az).T
            })
    return out


def same_shape(a, sa, b, sb, tol=0.1):
    from scipy.optimize import linear_sum_assignment
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if sorted(sa) != sorted(sb):
        return False
    total = 0.0
    for el in set(sa):
        ia = [i for i, s in enumerate(sa) if s == el]
        ib = [i for i, s in enumerate(sb) if s == el]
        d = np.linalg.norm(a[ia][:, None, :] - b[ib][None, :, :], axis=2)
        r, c = linear_sum_assignment(d)
        total += float(np.sum(d[r, c] ** 2))
    return (total / len(sa)) ** 0.5 < tol
