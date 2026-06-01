import warnings
from abc import abstractmethod
from contextlib import nullcontext
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Dict

import numpy as np
from clip_server.executors.helper import (
    preproc_image,
    preproc_text,
    set_rank,
    split_img_txt_da,
)
from clip_server.model import clip
from clip_server.model.tokenization import Tokenizer
from jina import DocumentArray, Executor, requests
from opentelemetry.trace import NoOpTracer


class BaseCLIPEncoder(Executor):
    """Template Method: 定义 encode/rank 骨架，子类填充模型构建与推理细节。

    Hook Points (子类必须实现):
        _build_model(name, **kwargs)   — 构建模型实例
        _encode_image_batch(data)      — 图像 minibatch 编码 + 后处理
        _encode_text_batch(data)       — 文本 minibatch 编码 + 后处理

    Hook Points (子类可选覆盖):
        RUNTIME                        — trace 标记 'torch'/'onnx'/'tensorrt'
        _return_np                     — 预处理是否返回 numpy (ONNX=True)
        _inference_context             — 推理上下文管理器 (torch=inference_mode)
        _resolve_device(device)        — 设备解析 (TensorRT 覆盖加入断言)
        _post_init(**kwargs)           — __init__ 末尾钩子 (ONNX session 启动等)
    """

    RUNTIME: str = None
    _return_np: bool = False

    def __init__(
        self,
        name: str = 'ViT-B-32::openai',
        device: str = None,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 32,
        access_paths: str = '@r',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._minibatch_size = minibatch_size
        self._access_paths = access_paths
        if 'traversal_paths' in kwargs:
            warnings.warn(
                '`traversal_paths` is deprecated. Use `access_paths` instead.'
            )
            self._access_paths = kwargs['traversal_paths']

        self._num_worker_preprocess = num_worker_preprocess
        self._pool = ThreadPool(processes=num_worker_preprocess)

        self._device = self._resolve_device(device)

        self._model = self._build_model(name, **kwargs)
        self._tokenizer = Tokenizer(name)
        self._image_transform = clip._transform_blob(self._model.image_size)

        if not self.tracer:
            self.tracer = NoOpTracer()

        self._post_init(**kwargs)

    # ================================================================
    #  子类必须实现
    # ================================================================

    @abstractmethod
    def _build_model(self, name: str, **kwargs):
        """构建并返回模型实例。"""

    @abstractmethod
    def _encode_image_batch(self, batch_data: dict) -> np.ndarray:
        """对单个 minibatch 执行图像编码，返回 numpy 数组。"""

    @abstractmethod
    def _encode_text_batch(self, batch_data: dict) -> np.ndarray:
        """对单个 minibatch 执行文本编码，返回 numpy 数组。"""

    # ================================================================
    #  子类可选覆盖
    # ================================================================

    @staticmethod
    def _resolve_device(device: str) -> str:
        """平台无关的设备解析。TensorRT 子类覆盖以加入 CUDA 断言。"""
        import torch

        return device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _post_init(self, **kwargs):
        """__init__ 末尾钩子。子类用于后置初始化（如 ONNX session 启动）。"""

    @property
    def _inference_context(self):
        """推理上下文管理器。Torch 子类覆盖为 torch.inference_mode()。"""
        return nullcontext()

    # ================================================================
    #  公共方法 — 模板方法（不再被子类重写）
    # ================================================================

    def _preproc_images(self, docs: DocumentArray, drop_image_content: bool):
        with self.monitor(
            name='preprocess_images_seconds',
            documentation='images preprocess time in seconds',
        ):
            with self.tracer.start_as_current_span('preprocess_images'):
                return preproc_image(
                    docs,
                    preprocess_fn=self._image_transform,
                    return_np=self._return_np,
                    drop_image_content=drop_image_content,
                    **self._preproc_image_kwargs,
                )

    def _preproc_texts(self, docs: DocumentArray):
        with self.monitor(
            name='preprocess_texts_seconds',
            documentation='texts preprocess time in seconds',
        ):
            with self.tracer.start_as_current_span('preprocess_texts'):
                return preproc_text(
                    docs,
                    tokenizer=self._tokenizer,
                    return_np=self._return_np,
                    **self._preproc_text_kwargs,
                )

    @requests(on='/rank')
    async def rank(self, docs: DocumentArray, parameters: Dict, **kwargs):
        _drop_image_content = parameters.get('drop_image_content', False)
        await self.encode(docs['@r,m'], drop_image_content=_drop_image_content)
        set_rank(docs)

    @requests
    async def encode(
        self,
        docs: DocumentArray,
        tracing_context=None,
        parameters: Dict = {},
        **kwargs,
    ):
        with self.tracer.start_as_current_span(
            'encode', context=tracing_context
        ) as span:
            span.set_attribute('device', self._device)
            span.set_attribute('runtime', self.RUNTIME)

            access_paths = parameters.get('access_paths', self._access_paths)
            if 'traversal_paths' in parameters:
                warnings.warn(
                    '`traversal_paths` is deprecated. Use `access_paths` instead.'
                )
                access_paths = parameters['traversal_paths']
            _drop_image_content = parameters.get('drop_image_content', False)

            _img_da = DocumentArray()
            _txt_da = DocumentArray()
            for d in docs[access_paths]:
                split_img_txt_da(d, _img_da, _txt_da)

            with self.tracer.start_as_current_span('inference') as inference_span:
                inference_span.set_attribute(
                    'drop_image_content', _drop_image_content
                )
                inference_span.set_attribute(
                    'minibatch_size', self._minibatch_size
                )
                inference_span.set_attribute('has_img_da', bool(_img_da))
                inference_span.set_attribute('has_txt_da', bool(_txt_da))

                with self._inference_context:
                    if _img_da:
                        with self.tracer.start_as_current_span(
                            'img_minibatch_encoding'
                        ) as img_encode_span:
                            img_encode_span.set_attribute(
                                'num_pool_workers', self._num_worker_preprocess
                            )
                            for minibatch, batch_data in _img_da.map_batch(
                                partial(
                                    self._preproc_images,
                                    drop_image_content=_drop_image_content,
                                ),
                                batch_size=self._minibatch_size,
                                pool=self._pool,
                            ):
                                with self.monitor(
                                    name='encode_images_seconds',
                                    documentation='images encode time in seconds',
                                ):
                                    minibatch.embeddings = (
                                        self._encode_image_batch(batch_data)
                                    )

                    if _txt_da:
                        with self.tracer.start_as_current_span(
                            'txt_minibatch_encoding'
                        ) as txt_encode_span:
                            txt_encode_span.set_attribute(
                                'num_pool_workers', self._num_worker_preprocess
                            )
                            for minibatch, batch_data in _txt_da.map_batch(
                                self._preproc_texts,
                                batch_size=self._minibatch_size,
                                pool=self._pool,
                            ):
                                with self.monitor(
                                    name='encode_texts_seconds',
                                    documentation='texts encode time in seconds',
                                ):
                                    minibatch.embeddings = (
                                        self._encode_text_batch(batch_data)
                                    )

        return docs

    def _cleanup_model(self):
        """子类可选覆盖：释放模型专属资源（ONNX session / TRT engine / CUDA cache）。"""

    async def close(self):
        try:
            self._cleanup_model()
        except Exception:
            pass

        if self._pool:
            self._pool.terminate()
            self._pool.join()

        await super().close()
