from __future__ import annotations
CH2O_TYPE2_REASON = 'Search avoids constraining binding to a single Cl placeholder. No weak-adsorption assumption for CH2O; stable parents may be reused.'
SPECIES = {
    'CH4': {
        'smiles': 'C',
        'mode': 'free:C',
        'references': ['C', 'COM'],
        'seed_families': ['one_H_down', 'two_H_down', 'three_H_down'],
        'placeholder': False
    },
    'CO2': {
        'smiles': 'O=C=O',
        'mode': 'free:C',
        'references': ['C', 'O1', 'O2', 'COM'],
        'seed_families': [
            'linear',
            'C_facing_bent',
            'O_facing_bent',
            'lying_multicontact'
        ],
        'O_C_O_angle_seed_deg': [180.0, 150.0, 130.0],
        'placeholder': False,
        'note': 'Angles initialize seeds and remain free during optimization. No arbitrary -1 charge or COClOH substitution.'
    },
    'CO': {
        'smiles': '[C-]#[O+]',
        'mode': 'free:C',
        'references': ['C', 'O', 'midpoint'],
        'seed_families': ['C_down', 'O_down', 'lying'],
        'placeholder': False,
        'note': 'Both C and O orientations are required; free:C alone does not establish C-down orientation.'
    },
    'H2O': {
        'smiles': 'O',
        'mode': 'free:O',
        'references': ['O', 'COM'],
        'seed_families': ['O_facing', 'H_facing', 'tilted', 'lying'],
        'placeholder': False
    },
    'H2': {
        'smiles': '[H][H]',
        'mode': 'free:H',
        'references': ['midpoint'],
        'seed_families': ['parallel', 'perpendicular', 'tilted'],
        'placeholder': False,
        'note': 'The midpoint adapter selects the midpoint; free:H selects the first H.'
    },
    'CH2O': {
        'smiles': 'C=O',
        'mode': 'free:C',
        'references': ['C', 'O', 'CO_midpoint'],
        'seed_families': ['C_facing', 'O_facing', 'lying_multicontact'],
        'placeholder': False,
        'default_type': 'T2',
        'routing_reason': CH2O_TYPE2_REASON
    },
    'CH3': {'smiles': 'ClC', 'mode': 'Cl', 'binder_element': 'C', 'placeholder': True},
    'CH2': {
        'smiles': 'ClCCl',
        'mode': 'Cl',
        'binder_element': 'C',
        'placeholder': True
    },
    'CH': {
        'smiles': 'ClC(Cl)Cl',
        'mode': 'Cl',
        'binder_element': 'C',
        'placeholder': True
    },
    'CHO': {
        'smiles': 'ClC=O',
        'mode': 'Cl',
        'binder_element': 'C',
        'placeholder': True
    },
    'OH': {'smiles': 'ClO', 'mode': 'Cl', 'binder_element': 'O', 'placeholder': True},
    'CH3O': {
        'smiles': 'COCl',
        'mode': 'Cl',
        'binder_element': 'O',
        'placeholder': True,
        'note': 'O binds the surface; C is the separate reaction anchor for C-H cleavage.'
    },
    'C': {'smiles': '[C]', 'mode': 'free:C', 'references': ['C'], 'placeholder': False},
    'O': {'smiles': '[O]', 'mode': 'free:O', 'references': ['O'], 'placeholder': False},
    'H': {'smiles': '[H]', 'mode': 'free:H', 'references': ['H'], 'placeholder': False}
}
SITE_SEEDS = ('ontop', 'bridge', 'fcc', 'hcp')
