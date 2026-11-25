from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from chavis.runtime.telemetry import StageClock, declared_cost_unknown


class StageTimer:

    def __init__(
        self,
        *,
        stage_id,
        attempt_id,
        factory_ref,
        input_artifact_ids,
        output_path,
        prep_origin,
        reused=False,
        historical_cost_unavailable_reason=None,
        fresh_factory_not_loaded=False
    ):
        self.output_path = Path(output_path)
        self.factory_ref = factory_ref
        self.historical_cost_unavailable_reason = historical_cost_unavailable_reason
        initial_factory = factory_ref()
        self.clock = StageClock.start(
            stage_id,
            attempt_id,
            SimpleNamespace(calls=0) if initial_factory is None and fresh_factory_not_loaded else initial_factory,
            input_artifact_ids=[value for value in input_artifact_ids if value is not None],
            prep_origin=prep_origin,
            reused=reused
        )
        self.record = {'status': 'running'}

    def finish(self, *, status, stop_reason=None, output_artifact_ids=()):
        self.record = self.clock.finish(
            self.factory_ref(),
            status=status,
            stop_reason=stop_reason,
            output_artifact_ids=output_artifact_ids,
            historical_cost_unavailable_reason=self.historical_cost_unavailable_reason
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + '.tmp')
        temporary.write_text(
            json.dumps(
                self.record,
                ensure_ascii=False,
                indent=2,
                allow_nan=False
            ) + '\n',
            encoding='utf-8'
        )
        temporary.replace(self.output_path)
        return self.record


def reused_saved_stage(reaction_id, entry):
    origin = entry.get('endpoint_preparation_origin') or {}
    reason = origin.get('historical_cost_unavailable_reason') or 'Historical endpoint preparation cost not recorded'
    row = declared_cost_unknown(
        stage_id='reuse_saved_endpoint_pair',
        attempt_id=reaction_id,
        prep_origin=origin.get('kind', 'reused_saved_pair'),
        reason=reason
    )
    digest = entry.get('source_sha256')
    row['input_artifact_ids'] = [digest] if digest else []
    row['output_artifact_ids'] = [digest] if digest else []
    return row
