import numpy as np
from ase.io import write
from ase.mep import NEB, DimerControl, MinModeAtoms, MinModeTranslate
from ase.optimize import FIRE
from ase.vibrations import Vibrations
from .geometry import from_ase, to_ase
from .ts_helpers import assert_slab, free_indices, initial_frames
from .scan_constraints import local_maxima
from .mic_idpp import PathPreparationError, prepare_idpp, mic_collision_pairs
from .policy import LEGACY_GUI_IDPP, NORMAL_REACTION_MIC_IDPP, NI_DRM_IDPP


class ScientificFailure(Exception):
    pass


class PreparationIncomplete(Exception):
    pass


class SearchMethods:
    route_idpp_policy = None

    def _idpp_policy(self):
        policy = self.route_idpp_policy
        if policy not in (NORMAL_REACTION_MIC_IDPP, NI_DRM_IDPP):
            raise ValueError('Search route IDPP policy was not selected')
        return policy

    def search_sequence(self, initial, final):
        if self._idpp_policy() != NI_DRM_IDPP:
            raise ValueError('Direct MIC sequence is not the legacy GUI route')
        result = {
            'method': 'LOCAL_MIC_IDPP_NEB',
            'target_found': False,
            'stages': [],
            'ts_candidates': []
        }
        self.report['search_sequence'] = result
        self.save()
        try:
            self.neb(initial, final, stop_after_neb=True)
            result['stages'].append({
                'method': 'LOCAL_MIC_IDPP_NEB',
                'stage': 'idpp',
                'status': 'converged' if self.report['idpp_preparation']['idpp']['converged'] else 'nonconverged_usable_band',
                'diagnostics': self.report['idpp_preparation']['idpp']
            })
            result['stages'].append({
                'method': 'LOCAL_MIC_IDPP_NEB',
                'stage': 'neb',
                'warm': self.report['neb']['warm'],
                'climbing': self.report['neb']['climbing'],
                'status': 'ci_converged' if self.report['neb']['climbing']['converged'] else 'ci_nonconverged_saved_candidate'
            })
            self.save()
        except ScientificFailure as exc:
            result['stages'].append({
                'method': 'LOCAL_MIC_IDPP_NEB',
                'stage': 'numerical_search',
                'status': 'nonconverged',
                'reason': str(exc)
            })
            self.save()
            return result
        except PathPreparationError as exc:
            self.report['idpp_preparation'] = exc.diagnostics
            result['stages'].append({
                'method': 'LOCAL_MIC_IDPP_NEB',
                'stage': 'idpp',
                'status': 'preparation_failure',
                'reason': str(exc)
            })
            self.save()
            return result
        try:
            ts = self.dimer_from_saved_band(initial, final)
            result['stages'].append({
                'method': 'SAVED_CANDIDATE_DIRECT_DIMER',
                'stage': 'dimer',
                'status': 'candidate_optimized',
                'ci_converged': self.report['neb']['climbing']['converged']
            })
            result['ts_candidates'].append({'snapshot': 'TS', 'ts': self.report['ts']})
        except ScientificFailure as exc:
            result['stages'].append({
                'method': 'SAVED_CANDIDATE_DIRECT_DIMER',
                'stage': 'dimer',
                'status': 'nonconverged',
                'reason': str(exc)
            })
            result['stages'].append({
                'stage': 'frequency',
                'status': 'not_run',
                'imaginary_count': None,
                'reason': 'no_converged_dimer_candidate'
            })
            self.save()
            return result
        try:
            mode = self.frequencies(ts)
        except ScientificFailure as exc:
            result['stages'].append({
                'stage': 'frequency',
                'status': 'not_single_imaginary',
                'reason': str(exc),
                'imaginary_count': self.report.get('frequencies', {}).get('imaginary_count')
            })
            self.save()
            return result
        result['stages'].append({'stage': 'frequency', 'status': 'one_imaginary'})
        try:
            self.connections(ts, mode, initial, final)
        except ScientificFailure as exc:
            result['stages'].append({
                'stage': 'downhill',
                'status': 'incomplete',
                'reason': str(exc)
            })
            self.save()
            return result
        checks = self.report.get('connections', {}).get('results', [])
        matched = False
        if len(checks) == 2:

            def match(row, name):
                record = row.get('matches', {}).get(name, {})
                return bool(row.get('converged') and record.get('edges_match') and (record.get('ads_rms_A', float('inf')) <= self.p['match_rms']) and (record.get('free_rms_A', float('inf')) <= self.p['match_rms']))
            matched = match(checks[0], 'IS') and match(checks[1], 'FS') or (match(checks[1], 'IS') and match(checks[0], 'FS'))
        result['target_found'] = bool(matched)
        result['stages'].append({
            'stage': 'downhill',
            'status': 'matched_both' if matched else 'connection_unresolved'
        })
        self.save()
        return result

    def neb(self, initial, final, stop_after_neb=False):
        policy = self._idpp_policy()
        self.check_endpoint_stationarity(initial, final)
        continuation = getattr(self, 'continuation', None)
        self.progress('Inspect stored CI-NEB path' if continuation else 'Initial IDPP path')
        if hasattr(self, 'band_frames'):
            frames = self.band_frames
            self.report['initial_path'] = getattr(
                self,
                'band_origin',
                'continuous scan branch (all scan constraints released)'
            )
        elif hasattr(self, 'seed_ts'):
            frames = initial_frames(
                initial,
                final,
                self.p['images'],
                self.seed_ts,
                policy=policy
            )
            self.report['initial_path'] = 'Apply IDPP separately from IS to previous TS and previous TS to FS. The TS initializes the path and remains unfixed during optimization.'
        else:
            frames, diagnostics = prepare_idpp(
                from_ase(initial),
                from_ase(final),
                self.p['images'],
                policy=policy
            )
            self.report['idpp_preparation'] = diagnostics
            self.report['initial_path'] = 'LOCAL_MIC_IDPP_NEB: ASE linear MIC + IDPP MIC'
            self.save()
        images = [to_ase(s) for s in frames]
        from fairchem.core import FAIRChemCalculator
        self.calculator()
        for a in images:
            a.calc = self.counted(FAIRChemCalculator(self._predictor, task_name='oc20'))
            assert_slab(a)
        write(
            self.d / ('continuation_start.extxyz' if continuation else 'idpp.extxyz'),
            images
        )
        neb = NEB(images, climb=bool(continuation), method='improvedtangent')
        if continuation and continuation.get('probe_only'):
            forces = neb.get_forces()
            energies = [float(a.get_potential_energy()) for a in images]
            raw_forces = [a.get_forces(apply_constraint=False).tolist() for a in images]
            if not np.isfinite(forces).all() or not np.isfinite(energies).all():
                raise ValueError('Nonfinite saved-band energy or force')
            from chavis.runtime.json_io import atomic
            atomic(
                self.d / 'continuation_probe_forces.json',
                {
                    'energies_eV': energies,
                    'raw_forces_eV_A': raw_forces,
                    'ci_projected_forces_eV_A': forces.tolist(),
                    'atom_ids': [s['atom_ids'] for s in frames],
                    'fixed_indices': [s['fixed_indices'] for s in frames]
                }
            )
            write(self.d / 'continuation_probe.extxyz', images)
            self.report['continuation_probe'] = {
                'energies_eV': energies,
                'ci_fmax': float(np.linalg.norm(forces, axis=1).max()),
                'optimizer_steps': 0,
                'endpoint_stationarity': self.report['endpoint_stationarity'],
                'all_images_recomputed': True,
                'coordinates_changed': False,
                'ts_or_frequency_validation': False
            }
            self.report['preparation_validation'] = {
                'stage': 'saved_CI_band_probe',
                'complete': True,
                'ts_search_run': False
            }
            self.artifacts['path'] = [from_ase(a) for a in images]
            self.save()
            return None

        def optimize(label, fmax, steps):
            opt = FIRE(
                neb,
                logfile=str(self.d / (label + '.log')),
                trajectory=str(self.d / (label + '.traj')),
                dt=0.05,
                maxstep=0.1
            )

            def record():
                self.check()
                if getattr(self, 'request', {}).get('record_neb_diagnostics'):
                    from .neb_diagnostics import record_neb
                    record_neb(self, opt, neb, images, label)
                if opt.nsteps % 10 == 0:
                    self.progress(
                        label,
                        step=opt.nsteps,
                        energies_eV=[float(a.get_potential_energy()) for a in images],
                        fmax=float(np.linalg.norm(neb.get_forces(), axis=1).max())
                    )
                    self.artifacts['path'] = [from_ase(a) for a in images]
                    self.save()
            opt.attach(record, interval=1)
            ok = bool(opt.run(fmax=fmax, steps=steps))
            return {
                'converged': ok,
                'steps': opt.nsteps,
                'fmax': float(np.linalg.norm(neb.get_forces(), axis=1).max())
            }
        if continuation:
            warm = {
                'skipped': True,
                'reason': 'saved climbing-image band; no repeated warm NEB'
            }
            steps = continuation['remaining_ci_steps']
            if not isinstance(steps, int) or steps <= 0 or steps + continuation['prior_spent_ci_steps'] > self.p['neb_steps']:
                raise ValueError('Invalid remaining CI step budget')
        else:
            warm = optimize('NEB', 0.1, self.p['neb_steps'])
            neb.climb = True
            steps = self.p['neb_steps']
        ci = optimize('CI_NEB', self.p['neb_fmax'], steps)
        if continuation:
            ci.update(
                segment_steps=ci['steps'],
                prior_spent_steps=continuation['prior_spent_ci_steps'],
                total_spent_steps=ci['steps'] + continuation['prior_spent_ci_steps'],
                checkpoint_step=continuation['checkpoint_step'],
                optimizer_history='fresh_FIRE'
            )
        energies = [float(a.get_potential_energy()) for a in images]
        peak = int(np.argmax(energies[1:-1])) + 1
        self.report['neb'] = {
            'warm': warm,
            'climbing': ci,
            'energies_eV': energies,
            'peak_image': peak
        }
        self.artifacts['path'] = [from_ase(a) for a in images]
        write(self.d / 'neb.extxyz', images)
        self.save()
        if not ci['converged'] and policy == LEGACY_GUI_IDPP:
            raise ScientificFailure('neb_step_limit')
        if not ci['converged']:
            self.report['neb']['candidate_rescue'] = {
                'method': 'SAVED_CANDIDATE_DIRECT_DIMER',
                'reason': 'CI_NEB_nonconverged',
                'ci_fmax_eV_A': ci['fmax'],
                'physical_convergence_not_claimed': True
            }
            self.save()
        if self.report.get('reaction', {}).get('category') == 'ADSORPTION':
            peaks = local_maxima(energies)
            self.report['neb']['local_peak_images'] = peaks
            if not peaks:
                self.report['path_observation'] = 'NO_TS_FOUND_ON_SAMPLED_PATH'
                self.save()
                raise ScientificFailure('no_saddle_on_sampled_path')
            peak = max(peaks, key=lambda i: energies[i])
            self.report['neb']['peak_image'] = peak
        if stop_after_neb:
            return None
        if policy == LEGACY_GUI_IDPP:
            ts = images[peak].copy()
            ts.calc = self.calculator()
            direction = images[peak + 1].positions - images[peak - 1].positions
            direction[assert_slab(ts)] = 0
            if np.linalg.norm(direction) < 1e-12:
                raise ScientificFailure('collapsed_band')
            direction /= np.linalg.norm(direction)
            return self._dimer_refine(
                ts,
                direction,
                initial,
                final,
                source_image=None,
                method=None,
                failure_reason='no_saddle_on_sampled_path'
            )
        return self.dimer_from_saved_band(initial, final)

    def dimer_from_saved_band(self, initial, final):
        if 'neb' not in self.report or 'path' not in self.artifacts:
            raise ScientificFailure('saved_neb_candidate_missing')
        peak = self.report['neb']['peak_image']
        frames = self.artifacts['path']
        if not 0 < peak < len(frames) - 1:
            raise ScientificFailure('saved_neb_peak_invalid')
        images = [to_ase(s) for s in frames]
        from ase.geometry import find_mic
        energies = self.report['neb']['energies_eV']
        ranked = sorted(range(1, len(images) - 1), key=lambda i: (-energies[i], i))
        candidates = []
        selected = None
        for index in ranked:
            probe = images[index]
            direction, _ = find_mic(
                images[index + 1].positions - images[index - 1].positions,
                probe.cell,
                probe.pbc
            )
            direction[assert_slab(probe)] = 0
            norm = float(np.linalg.norm(direction))
            collisions = mic_collision_pairs(
                from_ase(probe),
                probe.info['adsorbate_indices'],
                0.7
            )
            valid = np.isfinite(norm) and norm >= 1e-12 and (not collisions)
            candidates.append({
                'image': index,
                'energy_eV': energies[index],
                'direction_norm_A': norm,
                'mic_collision_pairs': collisions,
                'valid': bool(valid)
            })
            if valid:
                selected = (index, direction / norm)
                break
        self.report['neb']['candidate_selection'] = {
            'rank': 'interior energy descending, image index ascending',
            'evaluated': candidates,
            'selected_image': selected[0] if selected else None,
            'original_peak_image': peak
        }
        self.save()
        if selected is None:
            raise ScientificFailure('no_valid_saved_band_dimer_candidate')
        peak, direction = selected
        ts = images[peak].copy()
        ts.calc = self.calculator()
        return self._dimer_refine(
            ts,
            direction,
            initial,
            final,
            source_image=peak,
            method='SAVED_CANDIDATE_DIRECT_DIMER',
            failure_reason='dimer_nonconverged'
        )

    def _dimer_refine(
        self,
        ts,
        direction,
        initial,
        final,
        *,
        source_image,
        method,
        failure_reason
    ):
        self.progress('Dimer TS refinement')
        with DimerControl(
            logfile=str(self.d / 'dimer_control.log'),
            eigenmode_logfile=None,
            initial_eigenmode_method='displacement',
            displacement_method='vector',
            mask=[i in free_indices(ts) for i in range(len(ts))]
        ) as control:
            mma = MinModeAtoms(ts, control=control, eigenmodes=[direction])
            with MinModeTranslate(
                mma,
                logfile=str(self.d / 'dimer.log'),
                trajectory=str(self.d / 'dimer.traj')
            ) as opt:

                def record():
                    self.check()
                    if opt.nsteps % 10 == 0:
                        self.progress('Dimer', step=opt.nsteps)
                opt.attach(record, interval=1)
                ok = bool(opt.run(fmax=self.p['ts_fmax'], steps=self.p['dimer_steps']))
                steps = opt.nsteps
            curvature = float(mma.get_curvature())
        physical = float(np.linalg.norm(ts.get_forces(), axis=1).max())
        record = {
            'converged': bool(ok and physical <= self.p['ts_fmax'] and (curvature < 0)),
            'optimizer_converged': ok,
            'scan_constraints_released': True,
            'fmax': physical,
            'curvature': curvature,
            'steps': steps,
            'energy_eV': float(ts.get_potential_energy()),
            'barrier_forward_eV': float(ts.get_potential_energy() - initial.get_potential_energy()),
            'barrier_reverse_eV': float(ts.get_potential_energy() - final.get_potential_energy())
        }
        if method is not None:
            record.update(
                method=method,
                source_image=source_image,
                physical_fmax_target_eV_A=self.p['ts_fmax'],
                physical_fmax_ratio_to_target=physical / self.p['ts_fmax']
            )
        self.report['ts'] = record
        self.snapshot('TS', ts)
        if not ok or physical > self.p['ts_fmax'] or curvature >= 0:
            raise ScientificFailure('ts_step_limit' if steps >= self.p['dimer_steps'] else failure_reason)
        return ts

    def frequencies(self, ts):
        self.progress(
            'Vibrational analysis of all free atoms',
            free_atoms=len(free_indices(ts))
        )
        vib = Vibrations(
            ts,
            indices=free_indices(ts),
            name=str(self.d / 'vibrations' / 'ts'),
            delta=self.p['delta'],
            nfree=2
        )
        for disp, atoms in vib.iterdisplace(inplace=True):
            self.check()
            with vib.cache.lock(disp.name) as handle:
                if handle is not None:
                    handle.save(vib.calculate(atoms, disp))
            self.progress(
                'Vibrational displacement calculation',
                displacement=disp.name
            )
        vib.read()
        freq = vib.get_frequencies()
        imag = [i for i, f in enumerate(freq) if f.imag > 1e-06]
        all_modes = [vib.get_mode(i).tolist() for i in range(len(freq))]
        signed = [-float(f.imag) if f.imag > 0 else float(f.real) for f in freq]
        self.report['frequencies'] = {
            'cm-1': signed,
            'imaginary_modes': imag,
            'imaginary_count': len(imag),
            'indices': free_indices(ts),
            'delta_A': self.p['delta'],
            'nfree': 2,
            'scope': 'all_free_atoms_including_slab',
            'complete': True,
            'values_cm1': [[float(f.real), float(f.imag)] for f in freq],
            'imaginary_component_eps_cm1': 1e-06
        }
        self.artifacts['modes'] = all_modes
        vib.summary(log=str(self.d / 'frequencies.txt'))
        self.save()
        if len(imag) != 1:
            raise ScientificFailure('zero_imaginary' if not imag else 'multiple_imaginary')
        return np.array(all_modes[imag[0]])
