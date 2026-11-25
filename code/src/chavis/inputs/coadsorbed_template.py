import numpy as np
from chavis.search.geometry import from_ase, to_ase
from chavis.search.ts_helpers import ads_edges
from .type2_support import comparison_context


def mapped_cho_template(molecular, record):
    if record.get('model') != 'uma-s-1p2p1' or record.get('task') != 'oc20' or record.get('diagnostics', {}).get('converged') is not True:
        raise ValueError('CHO template needs explicit matching model/task and saved convergence')
    source = to_ase(record['structure'])
    spec = record['mapping']
    source_ids = source.arrays['rw_id'].tolist()
    target_ids = molecular.arrays['rw_id'].tolist()
    source_index = {v: i for i, v in enumerate(source_ids)}
    target_index = {v: i for i, v in enumerate(target_ids)}
    if len(source_index) != len(source) or len(target_index) != len(molecular):
        raise ValueError('Template atom IDs must be unique')
    source_ads = source.info['adsorbate_indices']
    target_ads = molecular.info['adsorbate_indices']
    if sorted((source[i].symbol for i in source_ads)) != ['C', 'H', 'O'] or sorted((molecular[i].symbol for i in target_ads)) != ['C', 'H', 'H', 'O']:
        raise ValueError('Template supports one CHO fragment plus one separate H only')
    source_slab = [i for i in range(len(source)) if i not in source_ads]
    target_slab = [i for i in range(len(molecular)) if i not in target_ads]
    sctx = comparison_context(from_ase(source), source.info.get('charge', 0))
    tctx = comparison_context(from_ase(molecular), molecular.info.get('charge', 0))
    for key in (
        'cell',
        'pbc',
        'fixed_indices',
        'fixed_positions',
        'model',
        'task',
        'charge',
        'constraints'
    ):
        if sctx[key] != tctx[key]:
            raise ValueError('CHO template context differs: ' + key)
    if [source_ids[i] for i in source_slab] != [target_ids[i] for i in target_slab] or any((source[i].symbol != 'Ni' for i in source_slab)) or any((molecular[i].symbol != 'Ni' for i in target_slab)):
        raise ValueError('CHO template must use the same single Ni slab IDs')
    mapping = {
        int(a): int(b) for a,
        b in spec['source_to_target_adsorbate_ids'].items()
    }
    if set(mapping) != {source_ids[i] for i in source_ads} or len(set(mapping.values())) != len(mapping) or set(mapping.values()) | {spec['separate_H_target_id']} != {target_ids[i] for i in target_ads}:
        raise ValueError('CHO template mapping does not cover the requested composition')
    pairs = [(source_index[a], target_index[b]) for a, b in mapping.items()]
    for i, j in [*zip(source_slab, target_slab), *pairs]:
        if source[i].symbol != molecular[j].symbol or source.get_initial_charges()[i] != molecular.get_initial_charges()[j] or source.get_initial_magnetic_moments()[i] != molecular.get_initial_magnetic_moments()[j]:
            raise ValueError('CHO template element/charge/magnetic conditions differ')
    mapped_edges = sorted((
        tuple(sorted((
            target_index[mapping[source_ids[a]]],
            target_index[mapping[source_ids[b]]]
        ))) for a,
        b in ads_edges(source)
    ))
    if mapped_edges != [tuple(e) for e in spec['expected_retained_edges']]:
        raise ValueError('Saved CHO template graph is incompatible')
    candidate = molecular.copy()
    candidate.positions[target_slab] = source.positions[source_slab]
    for i, j in pairs:
        candidate.positions[j] = source.positions[i]
    return candidate
