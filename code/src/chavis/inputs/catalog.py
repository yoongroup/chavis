from __future__ import annotations
from collections import Counter, deque
from copy import deepcopy
from chavis.runtime.hashing import sha256_file
import json
from pathlib import Path
from . import historical_sources
PRIMARY = tuple((f'D{i:02}' for i in range(1, 11))) + ('D13', 'N01', 'N02', 'N03', 'N04')
NEW = ('N01', 'N02', 'N03', 'N04')
PARENT_SITE = {'CHOH': 'bridge', 'COH': 'fcc'}
CL_COUNT = {'CHOH': 2, 'COH': 3}


def sha(path: Path) -> str:
    return sha256_file(path)


def _mapped_parent(smiles: str, expected_cl: int):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    mol = Chem.AddHs(mol)
    holders = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'Cl']
    if len(holders) != expected_cl:
        raise ValueError(f'Cl marker count {len(holders)} != {expected_cl}: {smiles}')
    binders = {n.GetIdx() for h in holders for n in mol.GetAtomWithIdx(h).GetNeighbors()}
    if len(binders) != 1 or mol.GetAtomWithIdx(next(iter(binders))).GetSymbol() != 'C':
        raise ValueError(f'Cl markers must all attach to the intended C: {smiles}')
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in holders]
    remap = {old: 36 + index for index, old in enumerate(keep)}
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in keep]
    edges = sorted([sorted((
        remap[b.GetBeginAtomIdx()],
        remap[b.GetEndAtomIdx()]
    )) for b in mol.GetBonds() if b.GetBeginAtomIdx() in remap and b.GetEndAtomIdx() in remap])
    return {
        'smiles_with_removable_Cl': smiles,
        'canonical_smiles_with_Cl': Chem.MolToSmiles(mol),
        'mode': 'Cl',
        'placeholder_indices_rdkit': holders,
        'binder_atom_id': remap[next(iter(binders))],
        'binder_element': 'C',
        'placeholder_count': len(holders),
        'fragment_symbols_after_Cl_removal': symbols,
        'fragment_atom_ids': list(range(36, 36 + len(symbols))),
        'formula_after_Cl_removal': dict(sorted(Counter(symbols).items())),
        'molecular_edges_after_Cl_removal': edges,
        'formal_charge_before_Cl_removal': Chem.GetFormalCharge(mol),
        'radical_electrons_before_Cl_removal': sum((a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    }


def _cut(mapped, elements):
    first, second = elements.split('-')
    symbols = {
        i: s for i,
        s in zip(
            mapped['fragment_atom_ids'],
            mapped['fragment_symbols_after_Cl_removal']
        )
    }
    candidates = [
        (i, j) for i,
        j in mapped['molecular_edges_after_Cl_removal'] if (symbols[i], symbols[j]) == (first, second)
    ]
    if len(candidates) != 1:
        raise ValueError(f'Expected one {elements} bond; got {candidates}')
    anchor, moving = candidates[0]
    graph = {i: set() for i in symbols}
    for i, j in mapped['molecular_edges_after_Cl_removal']:
        if (i, j) != (anchor, moving):
            graph[i].add(j)
            graph[j].add(i)
    fragment = {moving}
    pending = deque([moving])
    while pending:
        for j in graph[pending.popleft()] - fragment:
            fragment.add(j)
            pending.append(j)
    if anchor in fragment:
        raise ValueError('Target cut does not separate fragments')
    return (anchor, moving, [moving] + sorted(fragment - {moving}))


def make_catalog(*, campaign_root, science_root, species_config, reactions_config):
    species_config, reactions_config = (Path(species_config), Path(reactions_config))
    old = historical_sources.make_catalog(campaign_root, science_root)
    species = json.loads(species_config.read_text(encoding='utf-8'))
    reactions = {row['id']: row for row in json.loads(reactions_config.read_text(encoding='utf-8'))['reactions']}
    entries = {rid: deepcopy(old['entries'][rid]) for rid in PRIMARY if rid not in NEW}
    for rid, entry in entries.items():
        recorded = entry.get('endpoint_record') or {}
        anchor = entry.get('anchor') if type(entry.get('anchor')) is int else recorded.get('anchor')
        moving = entry.get('moving') if type(entry.get('moving')) is int else recorded.get('moving')
        if type(anchor) is not int or type(moving) is not int:
            raise ValueError(f'{rid}: historical endpoint lacks recorded target atom indices')
        is_struct = recorded.get('artifacts', {}).get('IS') or {}
        atom_ids = is_struct.get('atom_ids') or list(range(len(is_struct.get('symbols', []))))
        if not 0 <= anchor < len(atom_ids) or not 0 <= moving < len(atom_ids):
            raise ValueError(f'{rid}: target indices out of saved IS atom range')
        entry['anchor'] = anchor
        entry['moving'] = moving
        entry['moving_fragment'] = entry.get('moving_fragment') or recorded.get('moving_fragment')
        entry['target_pair_atom_ids'] = sorted((atom_ids[anchor], atom_ids[moving]))
        entry['requested_direction'] = 'dissociation'
        entry['queue_role'] = 'primary'
        entry['historical_endpoint_provenance_only'] = True
        entry['fresh_search_required'] = True
        entry['target_pair'] = sorted((anchor, moving))
        complete_saved_pair = rid not in ('D06', 'D10')
        entry['endpoint_preparation_origin'] = {
            'kind': 'reused_saved_pair' if complete_saved_pair else 'reused_saved_IS_FS_pending',
            'reused': True,
            'historical_cost_unavailable_reason': 'Historical endpoint preparation wall time and model-call count were not recorded in the selected source evidence',
            'historical_wall_s': None,
            'historical_model_calls': None
        }
    for rid in NEW:
        reaction = deepcopy(reactions[rid])
        parent = reaction['parent_species']
        recipe = species[parent]
        mapped = _mapped_parent(recipe['smiles'], CL_COUNT[parent])
        anchor, moving, moving_fragment = _cut(mapped, reaction['broken_bond_elements'])
        expected_is = mapped['molecular_edges_after_Cl_removal']
        target_pair = sorted((anchor, moving))
        expected_fs = [edge for edge in expected_is if edge != target_pair]
        if target_pair not in expected_is:
            raise AssertionError('Target pair absent from molecular graph')
        template = old['entries']['D07' if rid in ('N01', 'N02') else 'D10']
        settings = deepcopy(template.get('config', {}).get('settings') or {})
        if not settings:
            raise ValueError(f'No existing direct settings template for {rid}')
        settings['species'] = parent
        settings['site'] = 'fcc'
        requested = reaction['requested_direction']
        entries[rid] = {
            'reaction_id': rid,
            'route': 'direct_idpp_neb',
            'queue_role': 'primary',
            'requested_direction': requested,
            'computational_direction': 'dissociation',
            'barrier_reference': 'IS_energy' if requested == 'dissociation' else 'FS_energy',
            'species': parent,
            'parent_initial_site': PARENT_SITE[parent],
            'product_site_hypotheses': {
                'H': 'hollow',
                'CHO': 'C anchored; explicit eta2 later'
            } if rid == 'N01' else {
                'H': 'hollow',
                'CO': 'C anchored upright'
            } if rid == 'N02' else {'CH': 'hollow', 'OH': 'O hollow'} if rid == 'N03' else {'C': 'hollow', 'OH': 'O hollow'},
            'site_assertions_validated': False,
            'source_path': str(reactions_config),
            'source_sha256': sha(reactions_config),
            'input_path': None,
            'input_sha256': None,
            'source_run': None,
            'availability': 'pending_new_endpoint_preparation',
            'anchor': anchor,
            'moving': moving,
            'moving_fragment': moving_fragment,
            'target_pair': target_pair,
            'target_pair_atom_ids': target_pair,
            'target_bond_elements': reaction['broken_bond_elements'],
            'expected_IS_edges': expected_is,
            'expected_FS_edges': expected_fs,
            'r0_A': None,
            'r0_source': 'actual relaxed bound molecular IS MIC distance of target_pair; filled by prepare_new_endpoints.py',
            'molecular_template': mapped,
            'endpoint_record': {
                'reaction': reaction,
                'settings': settings,
                'model': 'uma-s-1p2p1',
                'task': 'oc20',
                'anchor': anchor,
                'moving': moving,
                'moving_fragment': moving_fragment,
                'expected_IS_edges': expected_is,
                'expected_FS_edges': expected_fs,
                'artifacts': {}
            },
            'is_structure': None,
            'config': {'settings': settings, 'model': 'uma-s-1p2p1', 'task': 'oc20'},
            'fresh_search_required': True,
            'endpoint_preparation_origin': {
                'kind': 'new_parent_and_FS_pending',
                'reused': False,
                'historical_cost_unavailable_reason': None
            }
        }
    if tuple(entries) != PRIMARY:
        raise AssertionError('Unexpected primary reaction order')
    refs = {aid: deepcopy(old['entries'][aid]) for aid in ('A01', 'A02', 'A03', 'A04', 'A05')}
    implementation = Path(historical_sources.__file__).resolve()
    catalog_implementation = Path(__file__).resolve()
    return {
        'schema_version': 2,
        'primary_ids': list(PRIMARY),
        'excluded_ids': ['D11', 'D12', 'X01', 'X02'],
        'reference_only_ids': list(refs),
        'entries': entries,
        'reference_entries': refs,
        'generated_from': {
            'endpoint_selector': str(implementation),
            'endpoint_selector_sha256': sha(implementation),
            'catalog_implementation': str(catalog_implementation),
            'catalog_implementation_sha256': sha(catalog_implementation),
            'campaign_root': str(campaign_root),
            'automation_science_root': str(science_root),
            'species_config': str(species_config),
            'species_config_sha256': sha(species_config),
            'reaction_config': str(reactions_config),
            'reaction_config_sha256': sha(reactions_config)
        }
    }
