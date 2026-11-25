from pathlib import Path
import hashlib
import time
from enum import Enum
REVISION = 'f611b917d9c68566bbbeccbb0aa0f7cad1696cb2'
ASSETS = {
    'checkpoints/uma-s-1p2p1.pt': 'b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce',
    'references/iso_atom_elem_refs.yaml': '8d6b57a33c6a139f80cd706f253a030bd9d93cbfb1bbe33fba066e23b7e8c29e',
    'references/form_elem_refs.yaml': '6de4f5105106bbb31fdc04b3fa215c50d379f043f3b6527e287475a3c2255c75'
}


def settings_json(value):
    if isinstance(value, dict):
        return {str(k): settings_json(v) for k, v in value.items()}
    if isinstance(value, set):
        return [settings_json(v) for v in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [settings_json(v) for v in value]
    if isinstance(value, Enum):
        return settings_json(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Unsupported inference setting: ' + type(value).__name__)


def assets():
    from huggingface_hub import hf_hub_download
    result = {}
    for name, digest in ASSETS.items():
        path = Path(hf_hub_download(
            'facebook/UMA',
            name,
            revision=REVISION,
            local_files_only=True
        ))
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError('Pinned model asset hash mismatch: ' + name)
        result[name] = {
            'path': str(path),
            'sha256': actual,
            'bytes': path.stat().st_size
        }
    return result


class ModelFactory:

    def __init__(self, check):
        import torch
        from omegaconf import OmegaConf
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        self.assets = assets()
        self.check = check
        self.calls = self.failed_calls = 0
        self.inference_seconds = 0.0
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('Validation requires the configured CUDA device')
        settings = InferenceSettings(
            compile=False,
            merge_mole=True,
            tf32=False,
            activation_checkpointing=False
        )
        started = time.monotonic()
        self.predictor = load_predict_unit(
            self.assets['checkpoints/uma-s-1p2p1.pt']['path'],
            settings,
            device='cuda',
            atom_refs=OmegaConf.load(self.assets['references/iso_atom_elem_refs.yaml']['path']),
            form_elem_refs=OmegaConf.load(self.assets['references/form_elem_refs.yaml']['path'])['refs'],
            workers=1,
            seed=41
        )
        self.load_seconds = time.monotonic() - started
        self.requested_settings = settings_json(settings.to_omegaconf())
        original = self.predictor.predict

        def counted(*args, **kwargs):
            self.check()
            begin = time.monotonic()
            self.calls += 1
            try:
                answer = original(*args, **kwargs)
                torch.cuda.synchronize()
                return answer
            except Exception:
                self.failed_calls += 1
                raise
            finally:
                self.inference_seconds += time.monotonic() - begin
        self.predictor.predict = counted

    def __call__(self):
        from fairchem.core import FAIRChemCalculator
        return FAIRChemCalculator(self.predictor, task_name='oc20')

    def metadata(self):
        resolved = getattr(self.predictor, 'inference_settings', None)
        resolved = settings_json(resolved.to_omegaconf()) if hasattr(resolved, 'to_omegaconf') else None
        return dict(
            model='uma-s-1p2p1',
            task='oc20',
            device='cuda',
            revision=REVISION,
            assets=self.assets,
            compile=False,
            merge_mole=True,
            tf32=False,
            activation_checkpointing=False,
            threads=1,
            workers=1,
            seed=41,
            requested_settings=self.requested_settings,
            resolved_settings=resolved,
            predictor_class=type(self.predictor).__name__,
            load_seconds=self.load_seconds,
            predict_calls=self.calls,
            failed_predict_calls=self.failed_calls,
            inference_seconds=self.inference_seconds,
            predict_call_definition='Actual MLIPPredictUnit.predict invocations, including failures'
        )
