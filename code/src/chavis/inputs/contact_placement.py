import math
from chavis.search.geometry import RADII, add, sub, mul, dot, norm


def contact_target(structure, moving, adsorbate, site_position, normal, scale=1.0):
    from chavis.search.collision import minimum_vector, dual as _dual
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Surface contact scale must be finite and greater than 0.')
    slab = [j for j in range(len(structure['symbols'])) if j not in set(adsorbate)]
    missing = sorted({structure['symbols'][i] for i in list(moving) + slab} - RADII.keys())
    if missing:
        raise ValueError('Missing contact radius: ' + ', '.join(missing) + '. No arbitrary radius substitution is made.')
    if not slab or not moving:
        raise ValueError('Drop placement requires both a moving fragment and a slab.')
    if abs(norm(normal) - 1) > 1e-08:
        raise ValueError('Contact normal must be a unit vector.')
    lead = structure['positions'][moving[0]]
    roots = []
    g0, g1, _ = _dual(structure)
    for i in moving:
        base = add(site_position, sub(structure['positions'][i], lead))
        for j in slab:
            raw = sub(base, structure['positions'][j])
            v = minimum_vector(raw, structure)
            z = dot(v, normal)
            plane = sub(v, mul(normal, z))
            rho = norm(plane)
            distance = scale * (RADII[structure['symbols'][i]] + RADII[structure['symbols'][j]])
            if rho >= distance:
                continue
            shift = sub(raw, v)
            roots.append(dict(
                shift_A=-z + math.sqrt((distance - rho) * (distance + rho)),
                moving_atom=i,
                slab_atom=j,
                distance_A=distance,
                planar_distance_A=rho,
                slab_cell_shift=[round(dot(shift, g0)), round(dot(shift, g1))]
            ))
    if not roots:
        raise ValueError('No slab contact at this lateral position. No fixed-height fallback is applied.')
    first = max(roots, key=lambda x: x['shift_A'])
    target = add(site_position, mul(normal, first['shift_A']))
    return (
        target,
        dict(
            contact_scale=scale,
            contact_scope='all_slab_atoms',
            height_from_site_A=first['shift_A'],
            normal_translation_A=dot(sub(target, lead), normal),
            site_position_A=list(site_position),
            first_contact=first,
            simultaneous_contacts=[x for x in roots if abs(x['shift_A'] - first['shift_A']) <= 1e-08]
        )
    )
