import os
import warnings
from typing import Optional, Union

import numpy as np
import torch
from clip_server.executors.base import BaseCLIPEncoder
from clip_server.helper import __cast_dtype__
from clip_server.model.clip_model import CLIPModel


class CLIPEncoder(BaseCLIPEncoder):
    RUNTIME = 'torch'
    _return_np = False

    def __init__(
        self,
        name: str = 'ViT-B-32::openai',
        device: Optional[str] = None,
        jit: bool = False,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 32,
        access_paths: str = '@r',
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(dtype, str):
            dtype = __cast_dtype__.get(dtype)
        elif not dtype:
            dtype = (
                torch.float32
                if device in ('cpu', torch.device('cpu'))
                else torch.float16
            )

        super().__init__(
            name=name,
            device=device,
            num_worker_preprocess=num_worker_preprocess,
            minibatch_size=minibatch_size,
            access_paths=access_paths,
            jit=jit,
            dtype=dtype,
            **kwargs,
        )

        self._preproc_image_kwargs = {'device': self._device, 'dtype': dtype}
        self._preproc_text_kwargs = {'device': self._device}

    def _post_init(self, **kwargs):
        if self._device.startswith('cuda'):
            return
        if 'OMP_NUM_THREADS' in os.environ:
            return
        if not hasattr(self.runtime_args, 'replicas'):
            return

        replicas = getattr(self.runtime_args, 'replicas', 1)
        num_threads = max(1, torch.get_num_threads() // replicas)
        if num_threads < 2:
            warnings.warn(
                f'Too many replicas ({replicas}) vs too few threads '
                f'{num_threads} may result in sub-optimal performance.'
            )

        torch.set_num_threads(max(num_threads, 1))
        torch.set_num_interop_threads(1)

    def _build_model(self, name: str, **kwargs):
        return CLIPModel.create(
            name,
            device=self._device,
            jit=kwargs.get('jit', False),
            dtype=kwargs.get('dtype'),
        )

    def _encode_image_batch(self, batch_data: dict) -> np.ndarray:
        return self._model.encode_image(**batch_data).cpu().numpy().astype(np.float32)

    def _encode_text_batch(self, batch_data: dict) -> np.ndarray:
        return self._model.encode_text(**batch_data).cpu().numpy().astype(np.float32)

    def _cleanup_model(self):
        import torch

        del self._model
        torch.cuda.empty_cache()

    @property
    def _inference_context(self):
        return torch.inference_mode()
