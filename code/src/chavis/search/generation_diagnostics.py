from .geometry import RADII, norm, sub, dot


def record_checks(structure, detail, params, reasons, collisions, normal):
    from .surface_sites import minimum_vector
    checks = []

    def add(code, label, actual, unit='', minimum=None, maximum=None, **extra):
        checks.append(dict(
            code=code,
            label=label,
            actual=actual,
            unit=unit,
            minimum=minimum,
            maximum=maximum,
            status='fail' if code in reasons else 'pass',
            **extra
        ))
    mode = detail['mode']
    if detail.get('contact'):
        pair = detail['contact']['first_contact']
        i = pair['moving_atom']
        j = pair['slab_atom']
        add(
            'contact_distance',
            f"Surface contact {structure['symbols'][i]}({i})–{structure['symbols'][j]}({j})",
            norm(minimum_vector(
                sub(structure['positions'][i], structure['positions'][j]),
                structure
            )),
            'Å',
            minimum=pair['distance_A'],
            maximum=pair['distance_A'],
            atoms=[i, j],
            scale=detail['contact']['contact_scale'],
            tolerance=1e-08
        )
    if 'distance_A' in detail:
        add(
            'radius',
            'Lateral distance' if detail['domain'] == 'sector' else '3-dimensional distance',
            detail['distance_A'],
            'Å',
            float(params.get('rmin', 1.8)),
            float(params.get('rmax', 3)),
            tolerance=1e-07
        )
        if mode == 'cleavage':
            add(
                'angle',
                'Lateral azimuth' if detail['domain'] == 'sector' else 'Bond-axis angle',
                detail['azimuth_offset_deg'] if detail['domain'] == 'sector' else detail['polar_deg'],
                '°',
                maximum=float(params.get('half_angle', 30)),
                tolerance=1e-07
            )
    scale = float(params.get('collision_scale', 0.7))
    add(
        'collision',
        'Colliding atom pairs',
        len(collisions),
        'pairs',
        maximum=0,
        scale=scale
    )
    for pair in collisions:
        i, j, distance = pair[:3]
        cutoff = scale * (RADII.get(structure['symbols'][i], 1.2) + RADII.get(structure['symbols'][j], 1.2))
        add(
            'collision',
            f"{structure['symbols'][i]}({i})–{structure['symbols'][j]}({j})",
            distance,
            'Å',
            minimum=cutoff,
            atoms=[i, j],
            scale=scale,
            periodic_image=len(pair) > 3,
            measurement_decimals=4,
            comparison='distance < cutoff',
            note='Distances display 4 decimal places; decisions use unrounded engine values.'
        )
    if mode == 'cleavage':
        i, j = (detail['moving'][0], detail['anchor'])
        if structure['symbols'][i] == structure['symbols'][j] == 'H':
            from chavis.judgment.policy import bond_threshold
            cutoff = bond_threshold('H', 'H')
        else:
            cutoff = float(params.get('bond_scale', 1.2)) * (RADII.get(structure['symbols'][i], 0.8) + RADII.get(structure['symbols'][j], 0.8))
        add(
            'target_bond_still_close',
            'Cleavage bond distance',
            norm(minimum_vector(
                sub(structure['positions'][i], structure['positions'][j]),
                structure
            )),
            'Å',
            minimum=cutoff,
            atoms=[i, j],
            comparison='distance < cutoff'
        )
    if structure['pbc'][2]:
        ads = set(structure['adsorbate_indices'])
        z = [
            dot(pos, normal) for i,
            pos in enumerate(structure['positions']) if i not in ads
        ]
        cell_z = abs(dot(structure['cell'][2], normal))
        low, high = (max(z) - cell_z + 1.5, min(z) + cell_z - 1.5)
        for i in detail['moving']:
            value = dot(structure['positions'][i], normal)
            add(
                'vacuum_image_too_close',
                f"{structure['symbols'][i]}({i}) vacuum-direction position",
                value,
                'Å',
                low,
                high,
                atoms=[i]
            )
            checks[-1]['status'] = 'fail' if value < low or value > high else 'pass'
    return checks
