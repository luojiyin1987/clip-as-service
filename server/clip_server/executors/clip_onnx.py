import os
import warnings
from typing import Optional

import onnxruntime as ort
from clip_server.executors.base import BaseCLIPEncoder
from clip_server.model.clip_onnx import CLIPOnnxModel


class CLIPEncoder(BaseCLIPEncoder):
    RUNTIME = 'onnx'
    _return_np = True

    def __init__(
        self,
        name: str = 'ViT-B-32::openai',
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 32,
        access_paths: str = '@r',
        model_path: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs,
    ):
        import torch

        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not dtype:
            dtype = 'fp32' if device in ('cpu', torch.device('cpu')) else 'fp16'

        super().__init__(
            name=name,
            device=device,
            num_worker_preprocess=num_worker_preprocess,
            minibatch_size=minibatch_size,
            access_paths=access_paths,
            model_path=model_path,
            dtype=dtype,
            **kwargs,
        )

        self._preproc_image_kwargs = {'dtype': dtype}
        self._preproc_text_kwargs = {}

    def _build_model(self, name: str, **kwargs):
        return CLIPOnnxModel(
            name, kwargs.get('model_path'), kwargs.get('dtype')
        )

    def _post_init(self, **kwargs):
        import torch

        providers = ['CPUExecutionProvider']
        if self._device.startswith('cuda'):
            providers.insert(0, 'CUDAExecutionProvider')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if not self._device.startswith('cuda') and (
            'OMP_NUM_THREADS' not in os.environ
            and hasattr(self.runtime_args, 'replicas')
        ):
            replicas = getattr(self.runtime_args, 'replicas', 1)
            num_threads = max(1, torch.get_num_threads() * 2 // replicas)
            if num_threads < 2:
                warnings.warn(
                    f'Too many replicas ({replicas}) vs too few threads '
                    f'{num_threads} may result in sub-optimal performance.'
                )

            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = max(num_threads, 1)

        self._model.start_sessions(
            sess_options=sess_options,
            providers=providers,
            dtype=kwargs.get('dtype'),
        )

    def _encode_image_batch(self, batch_data: dict):
        return self._model.encode_image(batch_data)

    def _encode_text_batch(self, batch_data: dict):
        return self._model.encode_text(batch_data)
