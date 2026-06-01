import numpy as np
from clip_server.executors.base import BaseCLIPEncoder
from clip_server.model.clip_trt import CLIPTensorRTModel


class CLIPEncoder(BaseCLIPEncoder):
    RUNTIME = 'tensorrt'
    _return_np = False

    def __init__(
        self,
        name: str = 'ViT-B-32::openai',
        device: str = 'cuda',
        num_worker_preprocess: int = 4,
        minibatch_size: int = 32,
        access_paths: str = '@r',
        **kwargs,
    ):
        self._preproc_image_kwargs = {'device': device}
        self._preproc_text_kwargs = {'device': device}

        super().__init__(
            name=name,
            device=device,
            num_worker_preprocess=num_worker_preprocess,
            minibatch_size=minibatch_size,
            access_paths=access_paths,
            **kwargs,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        assert device.startswith('cuda'), (
            f'Cannot perform inference on {device} with Nvidia TensorRT backend'
        )
        assert torch.cuda.is_available(), (
            'CUDA/GPU is not available on PyTorch. Please check CUDA installation'
        )
        return device

    def _build_model(self, name: str, **kwargs):
        return CLIPTensorRTModel(name)

    def _post_init(self, **kwargs):
        self._model.start_engines()

    def _encode_image_batch(self, batch_data: dict) -> np.ndarray:
        return (
            self._model.encode_image(batch_data)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _cleanup_model(self):
        import torch

        del self._model
        torch.cuda.empty_cache()

    def _encode_text_batch(self, batch_data: dict) -> np.ndarray:
        return (
            self._model.encode_text(batch_data)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
