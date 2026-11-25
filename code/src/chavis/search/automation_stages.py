from copy import deepcopy
from pathlib import Path
import hashlib
from chavis.runtime.hashing import sha256_file
import importlib.metadata
import json
import platform
import time
import numpy as np
from ase.constraints import FixAtoms, FixBondLengths
from ase.geometry import find_mic
from ase.optimize import BFGS
from .structures import load, dump, fixed_indices, assert_compatible, ModelViewCalculator
from .planning import method1_seeds, method2_seeds, scan_targets
from .generalization import band_cohorts, pilot_method2_seeds
from .judge import judge_dimer_step
from .return_policy import EndpointReturnMonitor
from chavis.judgment.native_automation import classify
from chavis.judgment.policy import bond_intact, H_H_PAIR_POLICY_KEY
from .policy import LEGACY_GUI_AUTOMATION, NI_DRM_AUTOMATION
from chavis.runtime.telemetry import DownhillFrameWriter
from .operational_limits import PILOT_LIMITS


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8'
    )
    temp.replace(path)


def pilot_context_sp(ctx, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = {}
    for label, source in (('IS', ctx.IS), ('FS', ctx.FS)):
        atoms = source.copy()
        atoms.info = deepcopy(source.info)
        before = dump(atoms)
        ctx.attach(atoms)
        energy = float(atoms.get_potential_energy())
        raw = np.asarray(atoms.get_forces(apply_constraint=False), float)
        if not np.isfinite(energy) or not np.isfinite(raw).all() or dump(atoms) != before:
            raise ValueError('Pilot fixed-geometry model context invalid or source mutated')
        rows[label] = dict(
            energy_eV=energy,
            saved_energy_eV=float(ctx.cfg.get('reference_energies', {}).get(
                label,
                ctx.saved_e_is if label == 'IS' else float('nan')
            )),
            raw_forces_eV_A=raw.tolist(),
            source_structure_sha256=hashlib.sha256(json.dumps(before, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            optimization_steps=0
        )
    for row in rows.values():
        if not np.isfinite(row['saved_energy_eV']):
            raise ValueError('Frozen saved endpoint energy unavailable')
        row['delta_fresh_minus_saved_eV'] = row['energy_eV'] - row['saved_energy_eV']
    result = dict(
        complete=True,
        IS=rows['IS'],
        FS=rows['FS'],
        energy_reference='fresh_SP_on_original_saved_IS_geometry',
        model_context=ctx.model_context,
        no_optimization=True
    )
    save(out / 'result.json', result)
    return result


class CalculationStopped(Exception):
    pass


class ScientificAbort(Exception):

    def __init__(self, code, measures):
        self.code, self.measures = (code, measures)
        super().__init__(code)


class Context:

    def __init__(
        self,
        config,
        endpoint_record,
        calculator_factory,
        *,
        check=None,
        model_context=None,
        reference_energy=None
    ):
        self.cfg = deepcopy(config)
        self.IS = load(endpoint_record['artifacts'][config['endpoints']['IS_artifact']])
        self.FS = load(endpoint_record['artifacts'][config['endpoints']['FS_artifact']])
        assert_compatible(self.IS, self.FS)
        self.fixed = fixed_indices(self.IS)
        from chavis.judgment.surface_context import selected_surface_element
        original = dump(self.IS)
        metal = selected_surface_element(original)
        slab = [i for i, atom in enumerate(self.IS) if atom.symbol == metal]
        if len(slab) != 36 or len(self.fixed) != 18 or set(self.fixed) != set(sorted(
            slab,
            key=lambda i: self.IS.positions[i, 2]
        )[:18]):
            raise ValueError('Expected validated 36-metal / bottom18 mask')
        self.surface_element = metal
        self.ni = slab
        self.free = [i for i in range(len(self.IS)) if i not in self.fixed]
        self.anchor, self.moving = (config['reaction'][k] for k in ('anchor', 'moving'))
        if self.anchor == self.moving or any((i not in self.IS.info['adsorbate_indices'] for i in (self.anchor, self.moving))):
            raise ValueError('Target IDs must be distinct adsorbate indices')
        self.z0 = float(np.sort(self.IS.positions[slab, 2])[-9:].mean())
        self.d_is = self.bond(self.IS)
        if config.get('generalization'):
            policy = config['generalization']
            if policy.get('version') != 'four-reaction-pilot-v1' or policy.get('original_saved_IS_r0_A') != self.d_is or policy.get('target_window_A') != [1.2 * self.d_is, 2.0 * self.d_is] or (policy.get('reaction_id') != config['reaction']['id']) or (policy.get('method') not in (1, 2)) or (policy.get('guard_mode') != 'shadow') or (policy.get('dimer_seed_cap') != 2) or (policy.get('science_seconds_cap') != 300) or (config.get('execution_limits') != PILOT_LIMITS):
                raise ValueError('Frozen pilot policy/r0 differs from original saved IS')
        self.saved_e_is = float(endpoint_record['IS']['energy_eV'])
        self.e_is = float(reference_energy if reference_energy is not None else self.saved_e_is)
        if not np.isfinite(self.e_is):
            raise ValueError('Finite saved-geometry IS energy required')
        self.energy_reference = 'fresh_SP_on_original_saved_IS_geometry' if reference_energy is not None else 'historical_saved_IS_energy'
        self.rules = {**config['judge'], 'minimum_approach_A': 0.1}
        self.cfg['judge'] = deepcopy(self.rules)
        self.calculator_factory = calculator_factory
        self.check = check or (lambda: None)
        self.model_context = deepcopy(model_context)
        self.calculators = []
        expected = {
            'endpoint_fmax': 0.01,
            'freq_delta_A': 0.01,
            'nfree': 2,
            'conn_disp_A': 0.2,
            'imag_eps_cm1': 1e-06
        }
        if any((config['science'].get(k) != v for k, v in expected.items())) or config['dimer'] != {
            'fmax': 0.005,
            'steps': 1500,
            'judge_every': 5
        }:
            raise ValueError('Validated scientific settings differ')

    def bond(self, atoms):
        return float(atoms.get_distance(self.anchor, self.moving, mic=True))

    def attach(self, atoms):
        self.check()
        atoms.calc = ModelViewCalculator(
            self.calculator_factory(),
            self.IS,
            check=self.check
        )
        self.calculators.append(atoms.calc)
        return atoms

    def structure_judge(self, atoms):
        return judge_dimer_step(
            atoms,
            atoms,
            self.IS,
            self.z0,
            self.rules,
            self.anchor,
            self.moving
        )

    def observe(self, optimizer):
        optimizer.attach(self.check, interval=1)


def variant(ctx, distance, relax, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    atoms = ctx.IS.copy()
    atoms.info = deepcopy(ctx.IS.info)
    vector, length = find_mic(
        ctx.IS.positions[ctx.anchor] - ctx.FS.positions[ctx.anchor],
        ctx.IS.cell,
        ctx.IS.pbc
    )
    if not np.isfinite(distance) or distance <= 0 or length <= 1e-12:
        raise ValueError('Invalid anchor translation')
    atoms.positions[atoms.info['adsorbate_indices']] += (distance - length) * vector / length
    result = dict(distance_A=distance, constrained_relax=bool(relax), relaxation=None)
    if relax:
        top = [i for i in ctx.ni if atoms.positions[i, 2] > ctx.z0 - 1.0]
        neighbors = sorted(
            top,
            key=lambda i: (atoms.get_distance(ctx.anchor, i, mic=True), i)
        )[:3]
        if len(neighbors) != 3:
            raise ValueError('Three top-layer metal atoms required')
        atoms.set_constraint([
            FixAtoms(indices=ctx.fixed),
            FixBondLengths([[ctx.anchor, i] for i in neighbors])
        ])
        ctx.attach(atoms)
        opt = BFGS(atoms, logfile=str(out / 'relax.log'), maxstep=0.15)
        ctx.observe(opt)
        ok = bool(opt.run(fmax=0.01, steps=ctx.cfg['science']['endpoint_steps']))
        result['relaxation'] = dict(
            converged=ok,
            steps=opt.nsteps,
            fixed_pairs=[[ctx.anchor, i] for i in neighbors],
            constrained_fmax=float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        )
        atoms.set_constraint(FixAtoms(indices=ctx.fixed))
    code, measures = ctx.structure_judge(atoms)
    result.update(structure=dump(atoms), judge=code, measures=measures)
    save(out / 'result.json', result)
    return (atoms, result)


def idpp_seeds(ctx, start, images, out, *, worker=None, pilot=False):
    if type(images) is not int or not 3 <= images <= 40:
        raise ValueError('IDPP image count must be 3..40')
    assert_compatible(ctx.IS, start)
    if worker is None:
        from worker import main as worker
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    frames = worker(dict(
        task='idpp',
        input=dump(start),
        output=dump(ctx.FS),
        images=images
    ))['frames']
    if len(frames) != images:
        raise ValueError('Incomplete IDPP band')
    band, energies = ([], [])
    for frame in frames:
        ctx.check()
        a = start.copy()
        a.info = deepcopy(start.info)
        a.positions[:] = np.asarray(frame['positions'], float)
        a.positions[ctx.fixed] = start.positions[ctx.fixed]
        band.append(a)
    if pilot:
        from .collision import collisions as _collisions
        diagnostics = []
        for i, a in enumerate(band):
            assert_compatible(ctx.IS, a)
            code, measures = ctx.structure_judge(a)
            collisions = _collisions(dump(a), a.info['adsorbate_indices'], 0.7)
            diagnostics.append(dict(
                index=i,
                judge=code,
                measures=measures,
                collisions=collisions,
                target_bond_A=ctx.bond(a)
            ))
        save(out / 'path_diagnostics.json', diagnostics)
        if any((row['judge'] != 'OK' or row['collisions'] for row in diagnostics)):
            raise CalculationStopped('PREPARATION_UNRESOLVED:PILOT_IDPP_PATH_GEOMETRY')
    for a in band:
        ctx.attach(a)
        energies.append(float(a.get_potential_energy()))
    selection = band_cohorts(
        energies,
        [ctx.bond(a) for a in band],
        ctx.d_is
    ) if pilot else method1_seeds(
        energies,
        [ctx.bond(a) for a in band],
        ctx.bond(start),
        ctx.rules['ref_bond_A']
    )
    result = dict(
        selection=selection,
        energies_eV=energies,
        energies_relative_variant_eV=[e - energies[0] for e in energies],
        energies_relative_saved_IS_eV=[e - ctx.e_is for e in energies],
        band=[dump(a) for a in band],
        ci_neb_required=False,
        path_diagnostics=diagnostics if pilot else None
    )
    save(out / 'result.json', result)
    return (band, result)


def bond_scan(ctx, out, *, pilot=False, resume_from=None, policy=NI_DRM_AUTOMATION):
    if policy not in (LEGACY_GUI_AUTOMATION, NI_DRM_AUTOMATION):
        raise ValueError('Unknown automation route policy')
    if resume_from is not None and (not policy.allow_partial_scan_resume):
        raise ValueError('Partial scan continuation is not enabled for this route')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    targets = scan_targets(ctx.bond(ctx.FS), ctx.d_is)
    points, frames, provenance = ([], [], None)
    if resume_from is not None:
        source = Path(resume_from).resolve()
        if not source.is_file() or source.name != 'scan.json':
            raise ValueError('Resume source must be a saved scan.json file')
        prior = json.loads(source.read_text(encoding='utf-8'))
        source_config = source.parent.parent / 'scan_config.json'
        source_reference = source.parent.parent / 'reference_energies.json'
        if not source_config.is_file() or not source_reference.is_file():
            raise ValueError('Resume source lacks scan config or fresh reference energies')
        old_config = json.loads(source_config.read_text(encoding='utf-8'))
        old_reference = json.loads(source_reference.read_text(encoding='utf-8'))
        if old_config != ctx.cfg:
            raise ValueError('Resume scan configuration differs from current context')
        saved_e_is = old_reference.get('IS_eV')
        if not isinstance(saved_e_is, (int, float)) or not np.isfinite(saved_e_is):
            raise ValueError('Resume source lacks finite fresh IS reference energy')
        fresh_difference = float(ctx.e_is - saved_e_is)
        if abs(fresh_difference) > 0.001:
            raise ValueError('Resume IS reference energy differs by more than 0.001 eV')
        if prior.get('complete') is not False or not isinstance(prior.get('points'), list) or (not isinstance(prior.get('frames'), list)):
            raise ValueError('Resume source is not a partial scan')
        if len(prior['points']) != len(prior['frames']) or not 0 < len(prior['points']) < len(targets):
            raise ValueError('Resume source has invalid scan prefix length')
        for k, (row, saved) in enumerate(zip(prior['points'], prior['frames'])):
            frame = load(saved)
            assert_compatible(ctx.IS, frame)
            if row.get('k') != k or not np.isclose(
                row.get('d_target_A'),
                targets[k],
                atol=1e-09,
                rtol=0
            ):
                raise ValueError(f'Resume scan target mismatch at point {k}')
            if not np.isclose(row.get('bond_A'), ctx.bond(frame), atol=1e-06, rtol=0):
                raise ValueError(f'Resume scan actual bond mismatch at point {k}')
            if not isinstance(row.get('E_rel_IS_eV'), (int, float)) or not np.isfinite(row['E_rel_IS_eV']):
                raise ValueError(f'Resume scan energy missing at point {k}')
            if k == 0 and (not np.allclose(
                frame.positions,
                ctx.FS.positions,
                atol=1e-06,
                rtol=0
            )):
                raise ValueError('Resume scan first point differs from current saved FS')
            points.append(deepcopy(row))
            frames.append(frame)
        provenance = {
            'source_scan_json': str(source),
            'source_scan_sha256': sha256_file(source),
            'source_scan_config_sha256': sha256_file(source_config),
            'source_reference_sha256': sha256_file(source_reference),
            'reused_prefix_points': len(points),
            'next_index': len(points),
            'source_IS_reference_eV': float(saved_e_is),
            'current_fresh_IS_reference_eV': float(ctx.e_is),
            'fresh_reference_difference_eV': fresh_difference,
            'remaining_original_targets_A': targets[len(points):],
            'source_evidence_preserved': True,
            'elapsed_s_excludes_interruption_gap': True
        }
        ctx.e_is = float(saved_e_is)
        ctx.energy_reference = 'resumed_original_fresh_SP_on_saved_IS_geometry'
        current = frames[-1].copy()
        current.info = deepcopy(frames[-1].info)
    else:
        current = ctx.FS.copy()
        current.info = deepcopy(ctx.FS.info)
    started = time.monotonic()
    elapsed_offset = float(points[-1].get('elapsed_s', 0.0)) if points else 0.0
    if provenance is not None:
        partial = dict(points=points, frames=[dump(a) for a in frames], complete=False)
        if policy.include_resume_provenance:
            partial['resume_provenance'] = provenance
        save(out / 'scan.json', partial)
    for k in range(len(points), len(targets)):
        target = targets[k]
        ctx.check()
        vector = current.get_distance(ctx.anchor, ctx.moving, mic=True, vector=True)
        length = np.linalg.norm(vector)
        if length <= 1e-12:
            raise ValueError('Zero target bond direction')
        current.set_constraint()
        current.positions[ctx.moving] = current.positions[ctx.anchor] + target * vector / length
        current.set_constraint([
            FixAtoms(indices=ctx.ni),
            FixBondLengths([[ctx.anchor, ctx.moving]])
        ])
        ctx.attach(current)
        opt = BFGS(current, logfile=str(out / f'pt{k:02d}.log'), maxstep=0.1)
        ctx.observe(opt)
        ok = bool(opt.run(fmax=0.05, steps=300))
        energy = float(current.get_potential_energy())
        restored = current.copy()
        restored.info = deepcopy(current.info)
        restored.set_constraint(FixAtoms(indices=ctx.fixed))
        code, measures = ctx.structure_judge(restored)
        frames.append(restored)
        points.append(dict(
            k=k,
            d_target_A=target,
            bond_A=ctx.bond(restored),
            converged=ok,
            steps=opt.nsteps,
            E_rel_IS_eV=energy - ctx.e_is,
            judge=code,
            measures=measures,
            elapsed_s=elapsed_offset + time.monotonic() - started
        ))
        partial = dict(points=points, frames=[dump(a) for a in frames], complete=False)
        if policy.include_resume_provenance:
            partial['resume_provenance'] = provenance
        save(out / 'scan.json', partial)
    selection = pilot_method2_seeds(
        [p['E_rel_IS_eV'] for p in points],
        [p['bond_A'] for p in points],
        ctx.d_is
    ) if pilot else method2_seeds(
        [p['E_rel_IS_eV'] for p in points],
        [p['bond_A'] for p in points],
        ctx.d_is
    )
    result = dict(
        points=points,
        frames=[dump(a) for a in frames],
        complete=True,
        selection=selection,
        scan_fmax=0.05,
        scan_steps=300,
        optimizer_fixed_indices=ctx.ni,
        model_fixed_indices=ctx.fixed
    )
    if policy.include_resume_provenance:
        result['resume_provenance'] = provenance
    save(out / 'scan.json', result)
    return (frames, result)


def tangent(frames, index):
    a, b = (frames[max(0, index - 1)], frames[min(len(frames) - 1, index + 1)])
    return find_mic(b.positions - a.positions, b.cell, b.pbc)[0]


def saved_endpoint_metrics(ctx, atoms, seed):
    ads = list(ctx.IS.info['adsorbate_indices'])
    free_ni = [i for i in ctx.ni if i not in ctx.fixed]
    result = {}
    for label, endpoint in (('IS', ctx.IS), ('FS', ctx.FS)):
        now_ads = find_mic(
            atoms.positions[ads] - endpoint.positions[ads],
            atoms.cell,
            atoms.pbc
        )[1]
        seed_ads = find_mic(
            seed.positions[ads] - endpoint.positions[ads],
            seed.cell,
            seed.pbc
        )[1]
        ni = find_mic(
            atoms.positions[free_ni] - endpoint.positions[free_ni],
            atoms.cell,
            atoms.pbc
        )[1]
        result[label] = dict(
            ads_rmsd_A=float(np.sqrt(np.mean(now_ads ** 2))),
            seed_ads_rmsd_A=float(np.sqrt(np.mean(seed_ads ** 2))),
            free_ni_rmsd_A=float(np.sqrt(np.mean(ni ** 2))) if len(ni) else 0.0,
            bond_difference_A=abs(ctx.bond(atoms) - ctx.bond(endpoint))
        )
    return result


def dimer_validate(ctx, seed, direction, out, *, policy=NI_DRM_AUTOMATION):
    if policy not in (LEGACY_GUI_AUTOMATION, NI_DRM_AUTOMATION):
        raise ValueError('Unknown automation route policy')
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
    from ase.vibrations import Vibrations
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(
        status='running',
        anchor=ctx.anchor,
        moving=ctx.moving,
        source_policy='saved-is-window-v1' if ctx.cfg.get('generalization') else 'approach-0.1-v1',
        d_IS_A=ctx.d_is,
        target_bond_intact_ratio=1.25,
        target_bond_elements=[ctx.IS[ctx.anchor].symbol, ctx.IS[ctx.moving].symbol],
        h_h_pair_policy_key=H_H_PAIR_POLICY_KEY,
        energy_reference=ctx.energy_reference,
        reference_IS_energy_eV=ctx.e_is,
        historical_saved_IS_energy_eV=ctx.saved_e_is
    )
    shared_search = Path(__file__).resolve().parent
    versions = {}
    for package in ('ase', 'numpy', 'scipy', 'torch', 'fairchem-core'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    save(
        out / 'run_meta.json',
        dict(
            model_context=ctx.model_context,
            settings=ctx.cfg,
            versions=versions,
            original_model_fixed_indices=ctx.fixed,
            python=platform.python_version(),
            source_sha256={p.name: sha256_file(p) for p in shared_search.glob('*.py')},
            imported_app_source_sha256={p.name: sha256_file(p) for p in (
                shared_search / 'gui_ts_run.py',
                shared_search / 'geometry.py'
            )},
            note=policy.run_meta_note
        )
    )
    save(out / 'result.json', result)
    first_calc = len(ctx.calculators)
    try:
        assert_compatible(ctx.IS, seed)
        ts = seed.copy()
        ts.info = deepcopy(seed.info)
        ts.set_constraint(FixAtoms(indices=ctx.fixed))
        ctx.attach(ts)
        vector = np.array(direction, dtype=float, copy=True)
        if vector.shape != (len(ts), 3) or not np.isfinite(vector).all():
            raise ValueError('Invalid Dimer direction')
        vector[ctx.fixed] = 0
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError('Zero free-atom Dimer direction')
        vector /= norm
        monitor = None
        if ctx.cfg.get('generalization') or policy.default_endpoint_guard_mode is not None:
            identity = hashlib.sha256(json.dumps(
                dict(config=ctx.cfg, IS=dump(ctx.IS), FS=dump(ctx.FS)),
                sort_keys=True,
                allow_nan=False
            ).encode()).hexdigest()
            guard_mode = ctx.cfg.get('generalization', {}).get(
                'guard_mode',
                policy.default_endpoint_guard_mode
            )
            monitor = EndpointReturnMonitor(
                r0_A=ctx.d_is,
                context_key=identity,
                mode=guard_mode
            )
            if policy == NI_DRM_AUTOMATION:
                result['endpoint_return_guard'] = {
                    'policy': 'saved-endpoint-return-v1',
                    'mode': guard_mode,
                    'context_key': identity,
                    'per_seed_only': True
                }
                save(out / 'result.json', result)
        with DimerControl(
            logfile=str(out / 'dimer_control.log'),
            eigenmode_logfile=None,
            initial_eigenmode_method='displacement',
            displacement_method='vector',
            mask=[i in ctx.free for i in range(len(ts))]
        ) as control:
            mma = MinModeAtoms(ts, control=control, eigenmodes=[vector])
            with MinModeTranslate(mma, logfile=str(out / 'dimer.log')) as opt:

                def check():
                    ctx.check()
                    if opt.nsteps % 5 == 0:
                        code, measures = judge_dimer_step(
                            ts,
                            seed,
                            ctx.IS,
                            ctx.z0,
                            ctx.rules,
                            ctx.anchor,
                            ctx.moving
                        )
                        with (out / 'judges.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps(
                                dict(step=opt.nsteps, code=code, **measures),
                                allow_nan=False
                            ) + '\n')
                        if code != 'OK':
                            raise ScientificAbort(code, measures)
                        if monitor is not None:
                            checkpoint = dump(ts)
                            raw = np.asarray(
                                ts.get_forces(apply_constraint=False),
                                float
                            )
                            raw_fmax = float(np.linalg.norm(raw[ctx.free], axis=1).max())
                            if not np.isfinite(raw).all() or not np.isfinite(raw_fmax):
                                raw_saved, raw_fmax = (None, None)
                            else:
                                raw_saved = raw.tolist()
                            step_curvature = float(mma.get_curvature()) if opt.nsteps > 0 else None
                            if step_curvature is not None and (not np.isfinite(step_curvature)):
                                step_curvature = None
                            observation = monitor.observe(
                                step=opt.nsteps,
                                context_key=monitor.context_key,
                                endpoint_metrics=saved_endpoint_metrics(ctx, ts, seed),
                                raw_free_fmax_eV_A=raw_fmax,
                                curvature_eV_A2=step_curvature,
                                coordinates_saved=True
                            )
                            with (out / 'endpoint_return.jsonl').open('a', encoding='utf-8') as f:
                                f.write(json.dumps(
                                    dict(
                                        observation=observation,
                                        structure=checkpoint,
                                        raw_physical_forces_eV_A=raw_saved,
                                        raw_free_fmax_eV_A=raw_fmax,
                                        curvature_eV_A2=step_curvature
                                    ),
                                    allow_nan=False
                                ) + '\n')
                            if observation['code'].startswith('ABORT_RETURN_'):
                                raise ScientificAbort(observation['code'], observation)
                opt.attach(check, interval=1)
                ok = bool(opt.run(fmax=0.005, steps=1500))
                steps = opt.nsteps
            curvature = float(mma.get_curvature())
        fmax = float(np.linalg.norm(ts.get_forces(), axis=1).max())
        result.update(
            dimer=dict(
                converged=ok and fmax <= 0.005 and (curvature < 0),
                optimizer_converged=ok,
                fmax=fmax,
                curvature=curvature,
                steps=steps
            ),
            TS=dump(ts),
            E_TS_minus_IS_eV=float(ts.get_potential_energy()) - ctx.e_is
        )
        code, measures = ctx.structure_judge(ts)
        result.update(ts_judge=code, ts_judge_measures=measures)
        if result['dimer']['converged'] and code == 'OK':
            vib = Vibrations(
                ts,
                indices=ctx.free,
                name=str(out / 'vib'),
                delta=0.01,
                nfree=2
            )
            vib.run()
            frequencies = vib.get_frequencies()
            signed = [-float(x.imag) if x.imag > 0 else float(x.real) for x in frequencies]
            imag = [i for i, x in enumerate(frequencies) if x.imag > 1e-06]
            result['frequencies'] = dict(
                cm1=signed,
                imaginary_modes=imag,
                imaginary_count=len(imag),
                n_free=len(ctx.free),
                indices=ctx.free,
                expected_free_indices=ctx.free,
                delta_A=0.01,
                nfree=2,
                completed=True
            )
            if len(imag) == 1:
                mode = vib.get_mode(imag[0])
                mode[ctx.fixed] = 0
                mode /= np.linalg.norm(mode, axis=1).max()
                np.save(out / 'imaginary_mode.npy', mode)
                connections = {}
                for sign, label in ((-1, 'minus'), (1, 'plus')):
                    ctx.check()
                    a = ts.copy()
                    a.info = deepcopy(ts.info)
                    a.positions += sign * 0.2 * mode
                    ctx.attach(a)
                    opt = BFGS(a, logfile=str(out / f'{label}.log'), maxstep=0.15)
                    ctx.observe(opt)
                    frame_path = out / f'{label}_downhill.extxyz'
                    frames = DownhillFrameWriter(
                        frame_path,
                        atom_ids=a.arrays['rw_id'].tolist(),
                        fixed_indices=ctx.fixed,
                        branch=label
                    )
                    opt.attach(frames.record, interval=1, atoms=a)
                    ok = bool(opt.run(
                        fmax=0.01,
                        steps=ctx.cfg['science']['endpoint_steps']
                    ))
                    c, m = ctx.structure_judge(a)
                    distance = ctx.bond(a)
                    connections[label] = dict(
                        converged=ok,
                        steps=opt.nsteps,
                        judge=c,
                        measures=m,
                        fmax=float(np.linalg.norm(a.get_forces(), axis=1).max()),
                        d_target_A=distance,
                        target_state='intact' if bond_intact(
                            distance,
                            ctx.IS[ctx.anchor].symbol,
                            ctx.IS[ctx.moving].symbol,
                            reference_length_A=ctx.d_is
                        ) else 'broken',
                        structure=dump(a),
                        energy_eV=float(a.get_potential_energy()),
                        downhill_frames_path=str(frame_path),
                        downhill_frame_count=frames.frames
                    )
                result['connections'] = connections
        result['status'] = 'done'
    except ScientificAbort as exc:
        result.update(
            status='aborted',
            abort=dict(code=exc.code, measures=exc.measures, step=opt.nsteps),
            aborted_structure=dump(ts)
        )
    except CalculationStopped as exc:
        result.update(
            status='interrupted',
            reason=str(exc),
            reason_category='OPERATOR_OR_RESOURCE_OR_BUDGET_STOP'
        )
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    result.update(
        classification=classify(result),
        elapsed_s=time.monotonic() - started,
        model_evaluations=sum((c.evaluations for c in ctx.calculators[first_calc:])),
        model_evaluations_definition='Adapter calculate calls; not guaranteed inner force evaluations',
        force_evaluations=None
    )
    save(out / 'result.json', result)
    return result
