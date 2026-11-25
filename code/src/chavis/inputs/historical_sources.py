from __future__ import annotations
from chavis.runtime.hashing import sha256_file
import json
import math
from pathlib import Path
DIRECT_IDS = (
    'D02',
    'D03',
    'D04',
    'D06',
    'D07',
    'D09',
    'D10',
    'D11',
    'D12',
    'X01',
    'X02'
)
ROUTE2_IDS = ('D01', 'D05', 'D08', 'D13')
ADSORPTION_PARENTS = {
    'A01': 'D01',
    'A02': 'D05',
    'A03': 'D09',
    'A04': 'D08',
    'A05': 'D13'
}


def sha(path: Path) -> str:
    return sha256_file(path)


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def usable_structure(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    symbols, positions = (obj.get('symbols'), obj.get('positions'))
    if not isinstance(symbols, list) or not isinstance(positions, list) or (not symbols) or (len(symbols) != len(positions)):
        return False
    try:
        return all((len(row) == 3 and all((math.isfinite(float(x)) for x in row)) for row in positions))
    except (TypeError, ValueError):
        return False


def compatible(a, b) -> bool:
    return usable_structure(a) and usable_structure(b) and (a['symbols'] == b['symbols']) and (a.get('atom_ids') == b.get('atom_ids')) and (a.get('fixed_indices') == b.get('fixed_indices'))


def campaign_records(campaign_root):
    rows = {}
    for path in sorted(
        Path(campaign_root).glob('*/attempts/*/result.json'),
        key=lambda p: p.as_posix()
    ):
        try:
            record = read(path)
            rid = (record.get('reaction') or {}).get('id')
            if rid and rid not in rows and compatible(
                record.get('artifacts', {}).get('IS'),
                record.get('artifacts', {}).get('FS')
            ):
                rows[rid] = (path, record)
        except (OSError, ValueError, TypeError):
            continue
    return rows


def partial_campaign_records(campaign_root):
    rows = {}
    candidates = {}
    for path in sorted(
        Path(campaign_root).glob('*/attempts/*/result.json'),
        key=lambda p: p.as_posix()
    ):
        try:
            record = read(path)
            rid = (record.get('reaction') or {}).get('id')
            is_struct = record.get('artifacts', {}).get('IS')
            if rid and usable_structure(is_struct):
                candidates.setdefault(rid, []).append((path, record))
        except (OSError, ValueError, TypeError):
            continue
    for rid, options in candidates.items():
        if rid == 'D06':

            def valid_bound_cho(item):
                _, record = item
                structure = record.get('artifacts', {}).get('IS', {})
                anchor, moving = (record.get('anchor'), record.get('moving'))
                pair = sorted((anchor, moving)) if isinstance(anchor, int) and isinstance(moving, int) else []
                expected = [sorted(edge) for edge in record.get('expected_IS_edges', [])]
                symbols, positions = (
                    structure.get('symbols', []),
                    structure.get('positions', [])
                )
                settings = record.get('settings', {})
                is_diag = record.get('IS', {})
                if not pair or pair not in expected or max(pair) >= len(symbols) or (symbols[anchor] != 'C') or (symbols[moving] != 'H') or (record.get('model') != 'uma-s-1p2p1') or (record.get('task') != 'oc20') or (record.get('IS', {}).get('converged') is not True) or (not math.isfinite(float(is_diag.get('fmax', math.inf)))) or (float(is_diag.get('fmax', math.inf)) > float(settings.get('endpoint_fmax', 0.01))) or (structure.get('fixed_indices') != list(range(18))) or (structure.get('adsorbate_indices') != [36, 37, 38]):
                    return False
                distance = math.dist(positions[anchor], positions[moving])
                return math.isfinite(distance) and distance < 1.284
            chosen = next(
                (item for item in options if valid_bound_cho(item)),
                options[0]
            )
            rows[rid] = chosen
        else:
            rows[rid] = options[0]
    return rows


def route2_records(science_root):
    rows = {}
    for path in sorted(
        Path(science_root).glob('*/inputs.json'),
        key=lambda p: p.as_posix()
    ):
        try:
            payload = read(path)
            generalization = payload.get('config', {}).get('generalization', {})
            rid = generalization.get('reaction_id')
            if rid in ROUTE2_IDS and rid not in rows and (generalization.get('method') == 2) and (generalization.get('guard_mode') == 'shadow'):
                ep = payload.get('endpoint_record', {})
                if compatible(
                    ep.get('artifacts', {}).get('IS'),
                    ep.get('artifacts', {}).get('FS')
                ):
                    rows[rid] = (path, payload)
        except (OSError, ValueError, TypeError):
            continue
    return rows


def make_catalog(campaign_root, science_root):
    campaign_root, science_root = (Path(campaign_root), Path(science_root))
    campaigns, partials, frozen = (
        campaign_records(campaign_root),
        partial_campaign_records(campaign_root),
        route2_records(science_root)
    )
    entries = {}
    for rid in DIRECT_IDS:
        if rid in campaigns:
            path, source = campaigns[rid]
            ep = source
            is_struct = source['artifacts']['IS']
            entries[rid] = {
                'reaction_id': rid,
                'route': 'direct_idpp_neb',
                'species': source.get('settings', {}).get('species'),
                'source_path': str(path),
                'source_sha256': sha(path),
                'input_path': None,
                'input_sha256': None,
                'source_run': path.parents[2].name,
                'availability': 'saved_IS_FS_pair',
                'endpoint_record': ep,
                'is_structure': is_struct,
                'config': {
                    'settings': source.get('settings'),
                    'model': source.get('model'),
                    'task': source.get('task')
                }
            }
        elif rid in partials:
            path, source = partials[rid]
            is_struct = source['artifacts']['IS']
            entries[rid] = {
                'reaction_id': rid,
                'route': 'direct_idpp_neb',
                'species': source.get('settings', {}).get('species'),
                'source_path': str(path),
                'source_sha256': sha(path),
                'input_path': None,
                'input_sha256': None,
                'source_run': path.parents[2].name,
                'availability': 'saved_valid_bound_IS_only_FS_missing' if rid == 'D06' else 'saved_IS_only_FS_missing',
                'endpoint_record': source,
                'is_structure': is_struct,
                'config': {
                    'settings': source.get('settings'),
                    'model': source.get('model'),
                    'task': source.get('task')
                },
                'anchor': source.get('anchor'),
                'moving': source.get('moving'),
                'moving_fragment': source.get('moving_fragment'),
                'expected_IS_edges': source.get('expected_IS_edges'),
                'expected_FS_edges': source.get('expected_FS_edges'),
                'preparation_source': {
                    'path': str(path),
                    'sha256': sha(path),
                    'available_artifacts': [k for k in ('parent_guess', 'IS', 'FS') if source.get('artifacts', {}).get(k) is not None],
                    'note': 'Saved, stationary bound CHO IS has the recorded C-H target edge; FS is absent.' if rid == 'D06' and source.get('IS', {}).get('converged') is True else 'Saved IS and parent guess are available; FS is absent. No FS endpoint is fabricated.'
                }
            }
        else:
            entries[rid] = {
                'reaction_id': rid,
                'route': 'direct_idpp_neb',
                'species': None,
                'source_path': None,
                'source_sha256': None,
                'input_path': None,
                'input_sha256': None,
                'source_run': None,
                'availability': 'input_missing_no_valid_saved_IS_FS_pair',
                'endpoint_record': None,
                'is_structure': None,
                'config': None
            }
    if 'D10' in campaigns and 'D06' in partials:
        old_path, old_record = campaigns['D10']
        is_path, is_record = partials['D06']
        is_structure = is_record['artifacts']['IS']
        new_record = {
            'reaction': old_record.get('reaction'),
            'settings': old_record.get('settings'),
            'model': is_record.get('model'),
            'task': is_record.get('task'),
            'anchor': 36,
            'moving': 37,
            'moving_fragment': [37],
            'expected_IS_edges': is_record.get('expected_IS_edges'),
            'expected_FS_edges': [[36, 38]],
            'IS': is_record.get('IS'),
            'artifacts': {'IS': is_structure}
        }
        entries['D10'] = {
            'reaction_id': 'D10',
            'route': 'direct_idpp_neb',
            'species': 'CHO',
            'source_path': str(is_path),
            'source_sha256': sha(is_path),
            'input_path': None,
            'input_sha256': None,
            'source_run': is_path.parents[2].name,
            'availability': 'saved_valid_bound_IS_only_FS_missing',
            'endpoint_record': new_record,
            'is_structure': is_structure,
            'config': {
                'settings': old_record.get('settings'),
                'model': is_record.get('model'),
                'task': is_record.get('task')
            },
            'anchor': 36,
            'moving': 37,
            'moving_fragment': [37],
            'expected_IS_edges': is_record.get('expected_IS_edges'),
            'expected_FS_edges': [[36, 38]],
            'endpoint_sources': {
                'IS': {
                    'path': str(is_path),
                    'sha256': sha(is_path),
                    'artifact': 'artifacts.IS',
                    'source_reaction_id': 'D06',
                    'note': 'Bound CHO IS with stationary same-context diagnostics; original atom IDs/mask/cell preserved.'
                },
                'FS': {
                    'status': 'missing',
                    'rejected_prior_path': str(old_path),
                    'rejected_prior_sha256': sha(old_path),
                    'rejected_prior_artifact': 'artifacts.FS',
                    'reason': 'Saved D10 FS fails target C-H retained graph; no old FS reused.'
                }
            }
        }
    for rid in ROUTE2_IDS:
        if rid in frozen:
            path, payload = frozen[rid]
            ep, cfg = (payload['endpoint_record'], payload['config'])
            entries[rid] = {
                'reaction_id': rid,
                'route': 'frozen_method2_scan_dimer',
                'species': ep.get('settings', {}).get('species'),
                'source_path': str(path),
                'source_sha256': sha(path),
                'input_path': str(path),
                'input_sha256': sha(path),
                'source_run': path.parent.name,
                'availability': 'frozen_inputs_and_saved_IS_FS_pair',
                'endpoint_record': ep,
                'is_structure': ep['artifacts']['IS'],
                'config': cfg
            }
        else:
            entries[rid] = {
                'reaction_id': rid,
                'route': 'frozen_method2_scan_dimer',
                'species': None,
                'source_path': None,
                'source_sha256': None,
                'input_path': None,
                'input_sha256': None,
                'source_run': None,
                'availability': 'input_missing_no_valid_frozen_method2_IS_FS_pair',
                'endpoint_record': None,
                'is_structure': None,
                'config': None
            }
    for aid, parent in ADSORPTION_PARENTS.items():
        src = entries.get(parent, {})
        is_struct = src.get('is_structure')
        ok = usable_structure(is_struct)
        entries[aid] = {
            'reaction_id': aid,
            'route': 'adsorption_z_scan',
            'species': {
                'A01': 'CH4',
                'A02': 'CO2',
                'A03': 'CO',
                'A04': 'H2O',
                'A05': 'H2'
            }[aid],
            'source_path': src.get('source_path'),
            'source_sha256': src.get('source_sha256'),
            'input_path': src.get('input_path'),
            'input_sha256': src.get('input_sha256'),
            'source_run': src.get('source_run'),
            'availability': 'saved_molecular_IS_only_no_FS' if ok else 'input_missing_no_saved_molecular_IS',
            'endpoint_record': {'artifacts': {'IS': is_struct}} if ok else None,
            'is_structure': is_struct if ok else None,
            'config': src.get('config'),
            'parent_reaction_id': parent
        }
    return {
        'schema_version': 1,
        'generated_from': {
            'campaign_root': str(campaign_root),
            'automation_science_root': str(science_root)
        },
        'entries': entries
    }
