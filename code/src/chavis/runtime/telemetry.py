from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import time
SCHEMA = 'chavis_stage_telemetry_v1'


def _calls(factory) -> int | None:
    value = getattr(factory, 'calls', None)
    return value if isinstance(value, int) and value >= 0 else None


@dataclass(frozen=True)
class StageClock:
    stage_id: str
    attempt_id: str
    started_at_utc: str
    start_wall_ns: int
    start_cpu_ns: int
    calls_before: int | None
    input_artifact_ids: tuple[str, ...]
    reused: bool
    prep_origin: str | None

    @classmethod
    def start(
        cls,
        stage_id: str,
        attempt_id: str,
        factory=None,
        *,
        input_artifact_ids=(),
        reused=False,
        prep_origin=None
    ) -> 'StageClock':
        if not stage_id or not attempt_id:
            raise ValueError('Stage and attempt IDs are required')
        return cls(
            stage_id,
            attempt_id,
            datetime.now(timezone.utc).isoformat(),
            time.monotonic_ns(),
            time.process_time_ns(),
            _calls(factory),
            tuple((str(value) for value in input_artifact_ids)),
            bool(reused),
            prep_origin
        )

    def finish(
        self,
        factory=None,
        *,
        status: str,
        stop_reason: str | None=None,
        output_artifact_ids=(),
        historical_cost_unavailable_reason=None
    ):
        end_wall_ns = time.monotonic_ns()
        end_cpu_ns = time.process_time_ns()
        after = _calls(factory)
        if self.calls_before is None or after is None:
            delta = None
            model_reason = historical_cost_unavailable_reason or 'model_counter_unavailable'
        else:
            delta = after - self.calls_before
            if delta < 0:
                raise ValueError('Model prediction counter decreased')
            model_reason = None
        if self.reused and delta is None and (not model_reason):
            model_reason = 'saved_endpoint_preparation_cost_not_recorded'
        wall = (end_wall_ns - self.start_wall_ns) / 1000000000.0
        cpu = (end_cpu_ns - self.start_cpu_ns) / 1000000000.0
        if not all((math.isfinite(v) and v >= 0 for v in (wall, cpu))):
            raise ValueError('Nonfinite or negative stage time')
        return {
            'schema': SCHEMA,
            'stage_id': self.stage_id,
            'attempt_id': self.attempt_id,
            'started_at_utc': self.started_at_utc,
            'ended_at_utc': datetime.now(timezone.utc).isoformat(),
            'wall_monotonic_s': wall,
            'process_cpu_s': cpu,
            'model_calls_before': self.calls_before,
            'model_calls_after': after,
            'model_calls_delta': delta,
            'model_calls_definition': 'actual_predict_invocations_not_cached_ASE_accessors',
            'model_calls_unavailable_reason': model_reason,
            'input_artifact_ids': list(self.input_artifact_ids),
            'output_artifact_ids': [str(value) for value in output_artifact_ids],
            'status': status,
            'stop_reason': stop_reason,
            'reused': self.reused,
            'prep_origin': self.prep_origin,
            'historical_cost_unavailable_reason': historical_cost_unavailable_reason,
            'model_load_s': None,
            'model_load_unavailable_reason': 'recorded_separately_by_model_factory'
        }


class DownhillFrameWriter:

    def __init__(self, path: str | Path, *, atom_ids, fixed_indices, branch: str):
        from ase.io import write
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.atom_ids = tuple((int(value) for value in atom_ids))
        self.fixed_indices = tuple(sorted(set((int(value) for value in fixed_indices))))
        self.branch = str(branch)
        self.frames = 0

    def record(self, atoms):
        from ase.io import write
        import numpy as np
        if len(atoms) != len(self.atom_ids):
            raise ValueError('Downhill frame atom inventory changed')
        if any((i < 0 or i >= len(atoms) for i in self.fixed_indices)):
            raise ValueError('Downhill frame fixed mask out of range')
        positions = np.asarray(atoms.positions, dtype=float)
        if positions.shape != (len(self.atom_ids), 3) or not np.isfinite(positions).all():
            raise ValueError('Downhill frame coordinates invalid')
        snapshot = atoms.copy()
        snapshot.calc = None
        snapshot.set_array('rw_id', np.asarray(self.atom_ids, dtype=int))
        snapshot.set_array(
            'rw_fixed',
            np.asarray([i in self.fixed_indices for i in range(len(atoms))], dtype=bool)
        )
        snapshot.info['mode_relax_branch'] = self.branch
        snapshot.info['mode_relax_frame_index'] = self.frames
        write(
            self.path,
            snapshot,
            format='extxyz',
            append=self.frames > 0,
            write_results=False
        )
        self.frames += 1


def declared_cost_unknown(
    *,
    stage_id: str,
    attempt_id: str,
    prep_origin: str,
    reason: str
):
    return {
        'schema': SCHEMA,
        'stage_id': stage_id,
        'attempt_id': attempt_id,
        'status': 'reused_saved_evidence',
        'stop_reason': None,
        'reused': True,
        'prep_origin': prep_origin,
        'wall_monotonic_s': None,
        'process_cpu_s': None,
        'model_calls_before': None,
        'model_calls_after': None,
        'model_calls_delta': None,
        'model_calls_unavailable_reason': reason,
        'historical_cost_unavailable_reason': reason,
        'model_load_s': None,
        'model_load_unavailable_reason': reason,
        'input_artifact_ids': [],
        'output_artifact_ids': []
    }
