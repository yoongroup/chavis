import numpy as np
from ase.constraints import FixAtoms
from chavis.search.geometry import from_ase, to_ase
from chavis.search.ts_helpers import ads_edges, assert_slab
from .type2_support import atomic, read, comparison_context
from .seed_geometry import cleavage, surface_frame
from .fs_sites import direct_fs_seeds


class Type2Methods:
    _preparation_incomplete = RuntimeError

    def define_reaction_atoms(self, molecular):
        anchor, moving, expected = cleavage(
            molecular,
            self.case,
            self.request['seed'].get('bond_variant', 0)
        )
        cut = tuple(sorted((anchor, moving[0])))
        self.report.update(
            anchor=anchor,
            moving=moving[0],
            moving_fragment=moving,
            expected_IS_edges=expected,
            expected_FS_edges=[e for e in expected if e != cut]
        )
        self.report['reaction_atom_ids'] = {
            'anchor': int(molecular.arrays['rw_id'][anchor]),
            'lead': int(molecular.arrays['rw_id'][moving[0]]),
            'moving_fragment': [int(molecular.arrays['rw_id'][i]) for i in moving]
        }
        from .generation_input import diagnose_generation_input
        self.report['molecular_cleavage_direction_diagnostic'] = diagnose_generation_input(
            from_ase(molecular),
            {
                'anchor': anchor,
                'moving': moving,
                'domain': 'sector',
                'half_angle': self.p['half_angle']
            }
        )

    def route_event(self, event, **evidence):
        self.report.setdefault('route_history', []).append({'event': event, **evidence})
        self.save()

    def validated_coadsorbed(self, candidate, label):
        final, diag = self.relax(candidate, label)
        edges = ads_edges(final)
        valid = diag['converged'] and edges == self.report['expected_FS_edges']
        self.report['fs_attempts'].append({
            'candidate': label,
            **diag,
            'edges': edges,
            'target_edges': self.report['expected_FS_edges'],
            'valid': valid
        })
        self.save()
        if valid:
            self.report['FS'] = diag
            self.report['chosen_candidate'] = label
            self.snapshot('FS', final)
            return final
        return None

    def direct_coadsorbed_final(self, molecular):
        self.define_reaction_atoms(molecular)
        template = self.request.get('coadsorbed_fragment_template')
        if template:
            from .coadsorbed_template import mapped_cho_template
            record = read(self.campaign / template)
            molecular = mapped_cho_template(molecular, record)
            self.snapshot('mapped_coadsorbed_initializer', molecular)
            self.route_event(
                'mapped_CHO_parent_initializer',
                source=record['source'],
                mapping=record['mapping'],
                source_slab_used_whole=True,
                target_atom_ids_unchanged=True
            )
        saved = self.request.get('coadsorbed_snapshot')
        if saved:
            record = read(self.campaign / saved)
            candidate = to_ase(record['structure'])
            context = lambda a: comparison_context(from_ase(a), a.info.get('charge', 0))
            if context(candidate) != context(molecular) or candidate.info['adsorbate_indices'] != molecular.info['adsorbate_indices'] or (not np.array_equal(
                candidate.get_initial_charges(),
                molecular.get_initial_charges()
            )):
                raise ValueError('Saved coadsorbed FS context/atom IDs differ')
            self.route_event(
                'reuse_saved_coadsorbed',
                source=record.get('source'),
                original_atom_ids_preserved=True
            )
            final = self.validated_coadsorbed(candidate, 'coadsorbed_revalidated')
            if final is not None:
                return final
            self.route_event('saved_coadsorbed_not_valid', fallback='direct_site_pairs')
        rows, metadata = direct_fs_seeds(
            molecular,
            self.report['moving_fragment'],
            self.report['anchor'],
            self.p,
            self.request['seed']['site']
        )
        atomic(
            self.d / 'direct_fs_seeds.json',
            {'metadata': metadata, 'candidates': rows}
        )
        self.route_event('direct_coadsorption', **metadata, enumerated=len(rows))
        accepted = [row for row in rows if row['accepted']]
        offset = self.request['seed'].get('candidate_offset', 0)
        if accepted:
            offset %= len(accepted)
            accepted = accepted[offset:] + accepted[:offset]
        for index, row in enumerate(accepted[:self.p['max_fs_trials']]):
            final = self.validated_coadsorbed(
                to_ase(row['structure']),
                f'direct_FS_trial_{index + 1}'
            )
            if final is not None:
                return final
        raise self._preparation_incomplete('direct_coadsorbed_FS_not_valid_in_declared_trials')

    def prepare_type2(self, molecular, provenance):
        self.route_event(
            'fs_first_entry',
            default_route=self.case.get('default_route'),
            requested_family=self.request['strategy_family'],
            stable_molecular_IS_required=False
        )
        if self.request.get('verified_endpoint_bundle'):
            return self.reuse_reviewed_endpoints(molecular)
        final = self.direct_coadsorbed_final(molecular)
        if self.request.get('preparation_stage') == 'coadsorbed_FS':
            self.report['preparation_validation'] = {
                'stage': 'coadsorbed_FS',
                'complete': True,
                'ts_search_run': False
            }
            self.save()
            return None
        family = self.request['strategy_family']
        if family in ('REVERSE_BOND_SCAN', 'COUPLED_Z_BOND_SCAN'):
            self.route_event(
                'reverse_bond_scan_started',
                pair_atom_ids=self.report['reaction_atom_ids']
            )
            return self.reverse_scan(final, molecular)
        anchor, lead = (self.report['anchor'], self.report['moving'])
        initial = final.copy()
        vector = initial.get_distance(anchor, lead, mic=True, vector=True)
        vector /= np.linalg.norm(vector)
        distance = molecular.get_distance(anchor, lead, mic=True)
        initial.positions[self.report['moving_fragment']] += initial.positions[anchor] + distance * vector - initial.positions[lead]
        self.snapshot('FS_aligned_molecular_seed', initial)
        relaxed, diag = self.relax(initial, 'IS')
        self.report['IS'] = diag
        if diag['converged'] and self.molecular_identity_accepted(relaxed):
            self.route_event('FS_aligned_molecular_IS_valid')
            return (relaxed, final)
        self.snapshot('FS_aligned_relaxed_arrival', relaxed)
        self.route_event(
            'FS_aligned_IS_changed',
            actual_edges=ads_edges(relaxed),
            converged=diag['converged'],
            fallback='REVERSE_BOND_SCAN'
        )
        if diag['converged'] and ads_edges(relaxed) == self.report['expected_FS_edges']:
            saved_final = self.validated_coadsorbed(relaxed, 'dissociated_coadsorbed_FS')
            if saved_final is not None:
                final = saved_final
        self.route_event(
            'reverse_bond_scan_started',
            pair_atom_ids=self.report['reaction_atom_ids']
        )
        return self.reverse_scan(final, molecular)
