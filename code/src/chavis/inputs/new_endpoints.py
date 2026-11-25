from __future__ import annotations
from copy import deepcopy
from chavis.runtime.hashing import sha256_file
import json
import math
from pathlib import Path
import traceback
from .telemetry import StageTimer
NEW = ('N01', 'N02', 'N03', 'N04')
PARENTS = {'CHOH': ('N01', 'N03'), 'COH': ('N02', 'N04')}
PARENT_SITES = {'CHOH': 'bridge', 'COH': 'fcc'}


def _sha(path):
    return sha256_file(path)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8'
    )
    temp.replace(path)


def _edges(atoms):
    from chavis.search.ts_helpers import ads_edges
    return sorted([sorted(e) for e in ads_edges(atoms)])


def _relax(seed, original, settings, output, name, factory, check):
    import numpy as np
    from ase.optimize import BFGS
    from chavis.search.structures import ModelViewCalculator
    from chavis.search.geometry import from_ase
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    work = seed.copy()
    work.calc = ModelViewCalculator(factory(), original, check=check)
    opt = BFGS(work, logfile=str(output / f'{name}.log'), maxstep=0.15)
    opt.attach(check, interval=1)
    converged = bool(opt.run(
        fmax=float(settings['endpoint_fmax']),
        steps=int(settings['endpoint_steps'])
    ))
    forces = np.asarray(work.get_forces(), float)
    energy = float(work.get_potential_energy())
    fmax = float(np.linalg.norm(forces, axis=1).max())
    finite = bool(math.isfinite(energy) and math.isfinite(fmax) and np.isfinite(work.positions).all())
    result = {
        'converged': converged,
        'steps': int(opt.nsteps),
        'fmax': fmax,
        'energy_eV': energy,
        'threshold_eV_A': float(settings['endpoint_fmax']),
        'finite': finite,
        'structure': from_ase(work),
        'adsorbate_edges': _edges(work),
        'model_context': factory.metadata(),
        'logfile': str(output / f'{name}.log')
    }
    _save(output / f'{name}.json', result)
    return (work, result)


def _prepare_parent(species, entries, template, output, get_factory, check):
    from ase import Atoms
    from fairchem.core import FAIRChemCalculator
    from geometry import from_ase
    from mother_pipeline import smiles_input, uma_engine
    from campaign_ni.seeds import orient, place_on_snapshot
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rid = PARENTS[species][0]
    entry = entries[rid]
    mapped = entry['molecular_template']
    settings = entry['config']['settings']
    fragment = smiles_input.build_fragment(
        species,
        mapped['smiles_with_removable_Cl'],
        'Cl'
    )
    if fragment.raw_symbols.count('Cl') != mapped['placeholder_count'] or fragment.raw_symbols[fragment.binder] != 'C':
        raise ValueError('Actual fragment builder disagrees with catalog Cl/C anchor')
    _save(
        output / 'gas_embed.json',
        {
            'symbols': fragment.raw_symbols,
            'positions': fragment.raw_positions,
            'charge': fragment.charge,
            'spin': fragment.spin,
            'binder_raw_index': fragment.binder,
            'holder_raw_indices': fragment.holders
        }
    )
    factory = get_factory()
    uma_engine._CALCS['omol'] = FAIRChemCalculator(factory.predictor, task_name='omol')
    uma_engine.CHECK_RUN = check
    try:
        gas = uma_engine.relax_gas(fragment)
    finally:
        if hasattr(uma_engine, 'CHECK_RUN'):
            del uma_engine.CHECK_RUN
    _save(
        output / 'gas_optimized.json',
        {
            'converged': gas['converged'],
            'steps': gas['steps'],
            'e_before': gas['e_before'],
            'e_after': gas['e_after'],
            'symbols': gas['atoms'].get_chemical_symbols(),
            'positions': gas['atoms'].positions.tolist(),
            'charge': fragment.charge,
            'spin': fragment.spin
        }
    )
    if not gas['converged']:
        raise ValueError('gas_uma_not_converged')
    smiles_input.finalize_direction(fragment)
    if fragment.symbols != mapped['fragment_symbols_after_Cl_removal'] or 'Cl' in fragment.symbols:
        raise ValueError('Post-Cl atom identity/order differs from RDKit catalog map')
    frag = Atoms(fragment.symbols, positions=fragment.positions)
    variant = {
        'orientation': 'placeholder_direction',
        'reference': 'C',
        'site': PARENT_SITES[species]
    }
    frag, weights = orient(frag, species, variant)
    parent_seed, placement = place_on_snapshot(
        template,
        frag,
        weights,
        PARENT_SITES[species]
    )
    parent_seed.info.update(charge=fragment.charge, spin=fragment.spin)
    if parent_seed.get_chemical_symbols()[36:] != mapped['fragment_symbols_after_Cl_removal']:
        raise ValueError('Placed parent atom order changed')
    _save(
        output / 'parent_seed.json',
        {
            'structure': from_ase(parent_seed),
            'placement': placement,
            'orientation': variant,
            'fragment_steps': fragment.steps,
            'site_is_hypothesis': True
        }
    )
    uma_engine._CALCS.clear()
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    parent, diag = _relax(
        parent_seed,
        parent_seed,
        settings,
        output,
        'molecular_IS',
        factory,
        check
    )
    expected = mapped['molecular_edges_after_Cl_removal']
    diag['target_edges'] = expected
    diag['graph_valid'] = diag['adsorbate_edges'] == expected
    diag['accepted_molecular_IS'] = bool(diag['converged'] and diag['finite'] and diag['graph_valid'])
    _save(output / 'molecular_IS_validation.json', diag)
    if not diag['accepted_molecular_IS']:
        raise ValueError('molecular_IS_not_converged_or_wrong_graph')
    return (parent, diag, factory)


def _prepare_fs(rid, entry, molecular, parent_diag, factory, output, check):
    from automation_search.structures import assert_compatible
    from campaign_ni.seeds import cleavage
    from campaign_ni.type2 import direct_fs_seeds
    from geometry import from_ase, to_ase
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = entry['config']['settings']
    anchor, moving, expected = cleavage(
        molecular,
        entry['endpoint_record']['reaction'],
        0
    )
    if [anchor, moving[0]] != [entry['anchor'], entry['moving']] or moving != entry['moving_fragment']:
        raise ValueError('Existing cleavage mapping disagrees with RDKit atom map')
    if sorted([list(e) for e in expected]) != entry['expected_IS_edges']:
        raise ValueError('Existing graph detector disagrees with mapped molecular graph')
    target_edges = entry['expected_FS_edges']
    rows, metadata = direct_fs_seeds(molecular, moving, anchor, settings, 'fcc')
    _save(
        output / 'fs_candidates.json',
        {
            'method': 'campaign_ni.type2.direct_fs_seeds',
            'generator': metadata,
            'candidate_count': len(rows),
            'candidates': rows,
            'product_site_hypotheses_only': entry['product_site_hypotheses']
        }
    )
    attempts = []
    chosen = None
    maximum = int(settings.get('max_fs_trials', 6))
    for index, row in enumerate(rows):
        item = {
            'candidate_index': index,
            'site': row['site'],
            'seed_accepted': bool(row['accepted']),
            'collision_pairs': row['collision_pairs'],
            'lateral_distance_A': row['lateral_distance_A'],
            'status': 'not_attempted'
        }
        if not row['accepted']:
            item['status'] = 'rejected_existing_collision_predicate'
        elif sum((v['status'] in (
            'accepted_FS',
            'relaxation_or_graph_rejected',
            'operational_error'
        ) for v in attempts)) >= maximum:
            item['status'] = 'not_attempted_declared_trial_limit'
        elif chosen is not None:
            item['status'] = 'not_attempted_first_valid_FS_selected'
        else:
            try:
                trial = to_ase(row['structure'])
                final, diag = _relax(
                    trial,
                    molecular,
                    settings,
                    output,
                    f'FS_trial_{index + 1:02}',
                    factory,
                    check
                )
                assert_compatible(molecular, final)
                valid = bool(diag['converged'] and diag['finite'] and (diag['adsorbate_edges'] == target_edges))
                item.update(
                    status='accepted_FS' if valid else 'relaxation_or_graph_rejected',
                    converged=diag['converged'],
                    fmax=diag['fmax'],
                    energy_eV=diag['energy_eV'],
                    actual_edges=diag['adsorbate_edges'],
                    target_edges=target_edges,
                    output_path=str(output / f'FS_trial_{index + 1:02}.json')
                )
                if valid:
                    chosen = (index, diag)
            except Exception as exc:
                item.update(
                    status='operational_error',
                    error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc()
                )
        attempts.append(item)
        _save(
            output / 'FS_preparation_progress.json',
            {
                'reaction_id': rid,
                'attempts': attempts,
                'selected_candidate_index': chosen[0] if chosen else None
            }
        )
    if chosen is None:
        raise ValueError('no_valid_FS_in_declared_trials')
    index, fs_diag = chosen
    updated = deepcopy(entry)
    record = updated['endpoint_record']
    record['artifacts'] = {'IS': parent_diag['structure'], 'FS': fs_diag['structure']}
    record['IS'] = {k: parent_diag[k] for k in (
        'converged',
        'steps',
        'fmax',
        'energy_eV'
    )}
    record['FS'] = {k: fs_diag[k] for k in ('converged', 'steps', 'fmax', 'energy_eV')}
    updated['is_structure'] = parent_diag['structure']
    updated['r0_A'] = float(molecular.get_distance(anchor, moving[0], mic=True))
    updated['r0_source'] = 'prepared molecular_IS; ASE get_distance(target_pair, mic=True)'
    updated['d_FS_A'] = float(to_ase(fs_diag['structure']).get_distance(anchor, moving[0], mic=True))
    updated['availability'] = 'prepared_new_IS_FS_pair'
    updated['preparation_diagnostics_path'] = str(output / 'FS_preparation_progress.json')
    updated['selected_fs_candidate_index'] = index
    updated['source_run'] = str(output.parent.name)
    updated['endpoint_sources'] = {
        'IS': {
            'path': str(output.parent / updated['species'] / 'molecular_IS.json'),
            'sha256': _sha(output.parent / updated['species'] / 'molecular_IS.json'),
            'artifact': 'structure'
        },
        'FS': {
            'path': str(output / f'FS_trial_{index + 1:02}.json'),
            'sha256': _sha(output / f'FS_trial_{index + 1:02}.json'),
            'artifact': 'structure'
        }
    }
    _save(output / 'prepared_entry.json', updated)
    return updated


def prepare_new_entries(catalog, out, factory=None, check=None):
    from geometry import to_ase
    from automation_search.uma_model import ModelFactory
    check = check or (lambda: None)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    updated = deepcopy(catalog)
    entries = updated['entries']
    template = to_ase(entries['D04']['endpoint_record']['artifacts']['IS'])
    holder = [factory]
    loading_species = [None]

    def get_factory():
        if holder[0] is None:
            species = loading_species[0]
            if species is None:
                raise RuntimeError('Model factory requested outside a preparation parent stage')
            parent_dir = output / species
            load_timer = StageTimer(
                stage_id='load_pinned_model',
                attempt_id=species,
                factory_ref=lambda: holder[0],
                input_artifact_ids=[],
                output_path=parent_dir / 'model_load_stage_telemetry.json',
                prep_origin='pinned_UMA_model_setup'
            )
            try:
                holder[0] = ModelFactory(check)
                _save(
                    parent_dir / 'model_load.json',
                    {
                        'model_context': holder[0].metadata(),
                        'load_seconds_separate_from_preparation_stage': holder[0].load_seconds,
                        'note': 'Model load begins after fragment/Cl validation; parent wall time includes this nested setup event, so do not sum both durations.'
                    }
                )
                load_timer.finish(
                    status='completed',
                    output_artifact_ids=[str(parent_dir / 'model_load.json')]
                )
            except Exception as exc:
                load_timer.finish(
                    status='operational_error',
                    stop_reason=f'{type(exc).__name__}: {exc}'
                )
                raise
        return holder[0]
    for species, ids in PARENTS.items():
        parent_dir = output / species
        loading_species[0] = species
        parent_timer = StageTimer(
            stage_id='prepare_new_molecular_parent',
            attempt_id=species,
            factory_ref=lambda: holder[0],
            input_artifact_ids=[
                entries[ids[0]].get('source_sha256'),
                entries['D04'].get('source_sha256')
            ],
            output_path=parent_dir / 'stage_telemetry.json',
            prep_origin='new_parent_from_removable_Cl_recipe',
            fresh_factory_not_loaded=True
        )
        try:
            parent, parent_diag, active_factory = _prepare_parent(
                species,
                entries,
                template,
                parent_dir,
                get_factory,
                check
            )
            parent_event = parent_timer.finish(
                status='completed',
                output_artifact_ids=[
                    str(parent_dir / 'molecular_IS_validation.json'),
                    str(parent_dir / 'gas_optimized.json')
                ]
            )
            for rid in ids:
                fs_dir = output / rid
                fs_timer = StageTimer(
                    stage_id='prepare_new_FS',
                    attempt_id=rid,
                    factory_ref=lambda: holder[0],
                    input_artifact_ids=[str(parent_dir / 'molecular_IS_validation.json')],
                    output_path=fs_dir / 'stage_telemetry.json',
                    prep_origin='new_FS_from_existing_type2_candidates'
                )
                try:
                    entries[rid] = _prepare_fs(
                        rid,
                        entries[rid],
                        parent,
                        parent_diag,
                        active_factory,
                        fs_dir,
                        check
                    )
                    fs_event = fs_timer.finish(
                        status='completed',
                        output_artifact_ids=[
                            str(fs_dir / 'prepared_entry.json'),
                            str(fs_dir / 'FS_preparation_progress.json')
                        ]
                    )
                except Exception as exc:
                    entries[rid]['availability'] = 'input_missing_new_FS_preparation_failed'
                    entries[rid]['preparation_error'] = f'{type(exc).__name__}: {exc}'
                    entries[rid]['preparation_diagnostics_path'] = str(fs_dir)
                    _save(
                        fs_dir / 'error.json',
                        {
                            'reaction_id': rid,
                            'error': entries[rid]['preparation_error'],
                            'traceback': traceback.format_exc()
                        }
                    )
                    fs_event = fs_timer.finish(
                        status='preparation_error',
                        stop_reason=entries[rid]['preparation_error'],
                        output_artifact_ids=[str(fs_dir / 'error.json')]
                    )
                entries[rid]['preparation_stage_telemetry'] = {
                    'model_load': str(parent_dir / 'model_load_stage_telemetry.json') if (parent_dir / 'model_load_stage_telemetry.json').is_file() else None,
                    'shared_parent': str(parent_dir / 'stage_telemetry.json'),
                    'reaction_FS': str(fs_dir / 'stage_telemetry.json'),
                    'parent_attempt_id': parent_event['attempt_id'],
                    'FS_attempt_id': fs_event['attempt_id']
                }
                _save(output / 'catalog_progress.json', updated)
        except Exception as exc:
            if parent_timer.record['status'] != 'running':
                raise
            error = f'{type(exc).__name__}: {exc}'
            for rid in ids:
                entries[rid]['availability'] = 'input_missing_new_parent_preparation_failed'
                entries[rid]['preparation_error'] = error
                entries[rid]['preparation_diagnostics_path'] = str(parent_dir)
                entries[rid]['preparation_stage_telemetry'] = {
                    'model_load': str(parent_dir / 'model_load_stage_telemetry.json') if (parent_dir / 'model_load_stage_telemetry.json').is_file() else None,
                    'shared_parent': str(parent_dir / 'stage_telemetry.json'),
                    'reaction_FS': None
                }
            _save(
                parent_dir / 'error.json',
                {
                    'species': species,
                    'reaction_ids': ids,
                    'error': error,
                    'traceback': traceback.format_exc()
                }
            )
            parent_timer.finish(
                status='preparation_error',
                stop_reason=error,
                output_artifact_ids=[str(parent_dir / 'error.json')]
            )
            _save(output / 'catalog_progress.json', updated)
    return updated
