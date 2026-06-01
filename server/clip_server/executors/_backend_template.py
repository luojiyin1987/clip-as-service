"""
模板示例：新增 CLIP 推理后端只需覆盖 3 个方法 + 设置 3 个属性。
以假设的「OpenVINO」后端为例。
"""

import numpy as np
from clip_server.executors.base import BaseCLIPEncoder


class CLIPEncoder(BaseCLIPEncoder):

    # ================================================================
    #  必须: 类属性
    # ================================================================
    RUNTIME = 'openvino'
    _return_np = True

    # ================================================================
    #  必须: __init__ — 解析参数 + 设置 preproc kwargs + 调用 super()
    # ================================================================
    def __init__(
        self,
        name: str = 'ViT-B-32::openai',
        device: str = 'cpu',
        num_worker_preprocess: int = 4,
        minibatch_size: int = 32,
        access_paths: str = '@r',
        **kwargs,
    ):
        # Step 1: 设置预处理参数（在 super().__init__ 前或后均可）
        self._preproc_image_kwargs = {'device': device}
        self._preproc_text_kwargs = {'device': device}

        # Step 2: 调用基类初始化
        super().__init__(
            name=name,
            device=device,
            num_worker_preprocess=num_worker_preprocess,
            minibatch_size=minibatch_size,
            access_paths=access_paths,
            **kwargs,
        )

    # ================================================================
    #  必须: 模型构建
    # ================================================================
    def _build_model(self, name: str, **kwargs):
        # return OpenVINOModel(name)   # ← 你的模型类
        raise NotImplementedError

    # ================================================================
    #  必须: 推理 + 后处理 — 接受 batch_data dict，返回 float32 numpy
    # ================================================================
    def _encode_image_batch(self, batch_data: dict) -> np.ndarray:
        # return self._model.encode_image(batch_data)     # ← 直接调模型
        raise NotImplementedError

    def _encode_text_batch(self, batch_data: dict) -> np.ndarray:
        # return self._model.encode_text(batch_data)      # ← 直接调模型
        raise NotImplementedError

    # ================================================================
    #  可选: 以下 hook 继承自基类，按需覆盖
    # ================================================================

    # def _post_init(self, **kwargs):
    #     """模型加载后的额外初始化。"""
    #     pass

    # def _cleanup_model(self):
    #     """释放模型资源。"""
    #     pass

    # @property
    # def _inference_context(self):
    #     """推理上下文管理器。"""
    #     from contextlib import nullcontext
    #     return nullcontext()
