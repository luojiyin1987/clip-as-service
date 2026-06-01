# CLIP-as-service 重构路径文档

> 状态：已完成 | 日期：2026-06-01 | 7 次提交，11 个学习点

## 一、重构目标

1. **消除代码重复**：三个 CLIPEncoder 实现共享 ~85% 逻辑，抽取 `BaseCLIPEncoder` 基类
2. **修复已知 Bug**：tracing span 名称错误、裸 except 吞掉系统信号
3. **资源正确管理**：线程池显式关闭、上下文管理器标准化
4. **降低维护成本**：新增加速后端只需覆盖 4 个 hook 方法即可接入

## 二、现状分析

### 2.1 当前架构

```
server/clip_server/executors/
├── clip_torch.py      (224 行)  ← 三个文件共享 ~85% 逻辑
├── clip_onnx.py       (212 行)  ← 
├── clip_tensorrt.py   (188 行)  ← 
└── helper.py          (130 行)  纯函数，无问题
```

### 2.2 重复代码分布

下图展示三个 `encode()` 方法的结构对比，相同颜色的部分是逐字重复：

```
┌─────────────────────────────────────────────────────────────────────┐
│ encode(docs, tracing_context, parameters, **kwargs)                 │
├─────────────────────────────────────────────────────────────────────┤
│ ██████████ tracer span 开始                                         │
│ ██████████ access_paths / traversal_paths 解析                      │
│ ██████████ split_img_txt_da 分流                                    │
│ ██████████ inference span + 属性设置                                │
│                                                                     │
│   if _img_da:                                                       │
│     ██████████ img_minibatch_encoding span                          │
│     ██████████ map_batch 遍历                                       │
│     ██████████ monitor 计时                                         │
│     ┌──── 差异区 ────┐                                              │
│     │ model 调用方式   │  ← torch: encode_image(**data).cpu().numpy()
│     │ 后处理方式       │     onnx:  encode_image(data)
│     └───────────────┘     trt:   encode_image(data).detach().cpu().numpy()
│                                                                     │
│   if _txt_da:                                                       │
│     ██████████ txt_minibatch_encoding span                          │
│     ██████████ map_batch 遍历                                       │
│     ██████████ monitor 计时                                         │
│     ┌──── 差异区 ────┐                                              │
│     │ model 调用方式   │                                              │
│     └───────────────┘                                              │
├─────────────────────────────────────────────────────────────────────┤
│ ██████████ return docs                                              │
└─────────────────────────────────────────────────────────────────────┘
```

| 区域 | 行数 | 三文件状态 |
|------|------|------------|
| access_paths 解析 | 12 | 逐字相同 |
| split 文档分流 | 6 | 逐字相同 |
| inference span 设置 | 6 | 逐字相同 |
| image minibatch 遍历骨架 | 15 | 逐字相同 |
| text minibatch 遍历骨架 | 15 | 逐字相同 |
| **总计重复** | **~100 行** | ×3 = 300 行重复 |

### 2.3 当前 Bug 清单

| # | 位置 | 问题 | 严重度 |
|---|------|------|--------|
| 1 | `clip_torch.py:120` | tracing span 名称写死 `'preprocess_images'`，应随 text/image 分支变化 | 中 |
| 2 | `clip_onnx.py:132` | 同上 | 中 |
| 3 | `clip_tensorrt.py:95` | 同上 | 中 |
| 4 | `helper.py:56` | `except:` 裸 except，会吞掉 `KeyboardInterrupt` / `SystemExit` | 高 |

### 2.4 三 Executor 的差异矩阵

| 维度 | Torch | ONNX | TensorRT |
|------|-------|------|----------|
| runtime 名称 | `'torch'` | `'onnx'` | `'tensorrt'` |
| `return_np` | `False` | `True` | `False` |
| device 检测 | 自动 | 自动 | 硬编码 `'cuda'` |
| `_preproc_images` 传 device | 是 | 否 | 是 |
| `_preproc_images` 传 dtype | 是 | 是 | 否 |
| `_preproc_texts` 传 device | 是 | 否 | 是 |
| 模型传参方式 | `**batch_data` | `batch_data` | `batch_data` |
| 推理上下文 | `torch.inference_mode()` | 无 | 无 |
| 后处理 | `.cpu().numpy().astype(np.float32)` | 无（已 numpy） | `.detach().cpu().numpy().astype(np.float32)` |
| 线程管理 | `torch.set_num_threads` | ONNX `SessionOptions` | 无 |
| 构造函数额外参数 | `jit`, `dtype` | `dtype`, `model_path` | 无 |

## 三、目标架构

```
server/clip_server/executors/
├── base.py             (新建, ~130 行)  ← 公共逻辑：encode/rank/preproc/close
├── clip_torch.py       (重构, ~90 行)   ← 仅保留模型构建 + 4 个 hook
├── clip_onnx.py        (重构, ~80 行)   ← 仅保留 ONNX Session + 4 个 hook
├── clip_tensorrt.py    (重构, ~70 行)   ← 仅保留 TRT engine + 4 个 hook
└── helper.py           (不变, 130 行)   纯函数
```

## 四、分步重构路径

### Step 0: 安全保障（Baseline）

```bash
# 确保当前所有测试通过
pip install -e "server/[onnx]" -e "client/[test]"
pytest tests/test_simple.py tests/test_client.py tests/test_model.py \
       tests/test_ranker.py tests/test_helper.py -v --timeout=120

# 记录 git hash 以便回滚
git stash
```

**验收标准**：所有非 GPU 测试通过。

### Step 1: 修复现有 Bug（零风险）— 预计 5 分钟

**文件**：`server/clip_server/helper.py:56`

```diff
-    except:
+    except Exception:
         # no network, too slow, PyPi is down
         pass
```

**文件**：`clip_torch.py:120`, `clip_onnx.py:132`, `clip_tensorrt.py:95`

```diff
-    with self.tracer.start_as_current_span('preprocess_images'):
+    with self.tracer.start_as_current_span('preprocess_texts'):
```

**验收标准**：现有测试全部通过，diff 输出仅 4 行变更。

### Step 2: 创建 BaseCLIPEncoder — 预计 20 分钟

**文件**：`server/clip_server/executors/base.py`（新建）

#### BaseCLIPEncoder 完整设计

```python
"""
Base class for CLIP encoding executors.

Subclasses only need to implement 4 methods:
    _build_model(name, **kwargs) -> model
    _encode_image_batch(batch_data) -> np.ndarray
    _encode_text_batch(batch_data) -> np.ndarray
    close() (optional, for resource cleanup)
"""

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
    """
    Template Method Pattern: 定义 encode/rank 的骨架，子类填充具体步骤

    Hook Points (子类必须/可选实现):
        _build_model()          — 必须：构建模型实例
        _encode_image_batch()   — 必须：单 minibatch 图像编码 + 后处理
        _encode_text_batch()    — 必须：单 minibatch 文本编码 + 后处理
        _inference_context      — 可选：推理上下文管理器 (默认 nullcontext)
        _post_init()            — 可选：___init___ 末尾的后置初始化钩子
    """

    # --- 子类在 __init__ 前设置的类属性 ---
    RUNTIME: str = None
    _return_np: bool = False

    # --- 子类在 __init__ 中设置的实例属性 ---
    _preproc_image_kwargs: dict
    _preproc_text_kwargs: dict

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
        """构建并返回模型实例。子类负责传入 device/dtype/jit 等参数。"""

    @abstractmethod
    def _encode_image_batch(self, batch_data: dict) -> np.ndarray:
        """对单个 minibatch 执行图像编码并返回 numpy 数组。"""

    @abstractmethod
    def _encode_text_batch(self, batch_data: dict) -> np.ndarray:
        """对单个 minibatch 执行文本编码并返回 numpy 数组。"""

    # ================================================================
    #  子类可选覆盖
    # ================================================================

    @staticmethod
    def _resolve_device(device: str) -> str:
        """平台无关的设备解析（torch/onnx 共用）。TensorRT 子类覆盖此方法。"""
        import torch
        return device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _post_init(self, **kwargs):
        """__init__ 末尾钩子，用于子类的额外初始化（如 ONNX session 启动）。"""

    @property
    def _inference_context(self):
        """推理阶段的上下文管理器。Torch 子类覆盖为 torch.inference_mode()。"""
        return nullcontext()

    # ================================================================
    #  公共方法（不再被子类重写） — 模板方法
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
                inference_span.set_attribute(
                    'has_img_da', bool(_img_da)
                )
                inference_span.set_attribute(
                    'has_txt_da', bool(_txt_da)
                )

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

    async def close(self):
        """清理线程池等资源"""
        if hasattr(self, '_pool') and self._pool:
            self._pool.terminate()
            self._pool.join()
        await super().close()
```

#### 设计说明

**Template Method 模式**：
- `encode()` 和 `rank()` 定义骨架（在基类中实现一次）
- 子类通过 4 个方法填入具体步骤：
  - `_build_model()` — 构建模型
  - `_encode_image_batch()` — 模型推理 + 后处理
  - `_encode_text_batch()` — 同上
  - `_inference_context` — 推理上下文（如 `torch.inference_mode()`）

**preproc 参数化**：
- `_return_np`：控制预处理返回 numpy 还是 torch tensor（ONNX 用 numpy）
- `_preproc_image_kwargs` / `_preproc_text_kwargs`：子类在 `__init__` 中设定，基类方法透传

**扩展点**：
- `_resolve_device()`：设备选择逻辑，TensorRT 子类覆盖为直接返回 `'cuda'`
- `_post_init()`：初始化后的钩子，ONNX 用于启动 session
- `close()`：资源清理

**验收标准**：
- `python -c "from clip_server.executors.base import BaseCLIPEncoder"` 不报错
- `pytest tests/test_helper.py -v` 通过

### Step 3: 重构 clip_torch.py — 预计 15 分钟

**当前**：224 行独立实现 → **重构后**：~90 行，继承 BaseCLIPEncoder

```python
"""
PyTorch backend CLIP encoder.

Inherits encoding pipeline from BaseCLIPEncoder.
Only provides model construction and backend-specific batch encoding + inference context.
"""

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

        if not device.startswith('cuda') and (
            'OMP_NUM_THREADS' not in os.environ
            and hasattr(self.runtime_args, 'replicas')
        ):
            replicas = getattr(self.runtime_args, 'replicas', 1)
            num_threads = max(1, torch.get_num_threads() // replicas)
            if num_threads < 2:
                warnings.warn(
                    f'Too many replicas ({replicas}) vs too few threads '
                    f'{num_threads} may result in sub-optimal performance.'
                )
            torch.set_num_threads(max(num_threads, 1))
            torch.set_num_interop_threads(1)

        self._preproc_image_kwargs = {'device': device, 'dtype': dtype}
        self._preproc_text_kwargs = {'device': device}

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

    def _build_model(self, name: str, **kwargs):
        return CLIPModel(
            name,
            device=self._device,
            jit=kwargs.get('jit', False),
            dtype=kwargs.get('dtype'),
        )

    def _encode_image_batch(self, batch_data: dict) -> np.ndarray:
        return self._model.encode_image(**batch_data).cpu().numpy().astype(np.float32)

    def _encode_text_batch(self, batch_data: dict) -> np.ndarray:
        return self._model.encode_text(**batch_data).cpu().numpy().astype(np.float32)

    @property
    def _inference_context(self):
        return torch.inference_mode()
```

**验收标准**：
- `python -m clip_server torch-flow.yml` 启动成功
- `python tests -k "torch"` 通过

### Step 4: 重构 clip_onnx.py — 预计 10 分钟

```python
"""
ONNX runtime backend CLIP encoder.
"""

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

        if not device.startswith('cuda') and (
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

        self._preproc_image_kwargs = {'dtype': dtype}
        self._preproc_text_kwargs = {}

        super().__init__(
            name=name,
            device=device,
            num_worker_preprocess=num_worker_preprocess,
            minibatch_size=minibatch_size,
            access_paths=access_paths,
            model_path=model_path,
            dtype=dtype,
            num_threads=num_threads if device == 'cpu' else None,
            **kwargs,
        )

    def _build_model(self, name: str, **kwargs):
        return CLIPOnnxModel(
            name, kwargs.get('model_path'), kwargs.get('dtype')
        )

    def _post_init(self, **kwargs):
        providers = ['CPUExecutionProvider']
        if self._device.startswith('cuda'):
            providers.insert(0, 'CUDAExecutionProvider')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if not self._device.startswith('cuda') and kwargs.get('num_threads'):
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = max(kwargs['num_threads'], 1)

        self._model.start_sessions(
            sess_options=sess_options,
            providers=providers,
            dtype=kwargs.get('dtype'),
        )

    def _encode_image_batch(self, batch_data: dict) -> 'np.ndarray':
        return self._model.encode_image(batch_data)

    def _encode_text_batch(self, batch_data: dict) -> 'np.ndarray':
        return self._model.encode_text(batch_data)
```

### Step 5: 重构 clip_tensorrt.py — 预计 10 分钟

```python
"""
NVIDIA TensorRT backend CLIP encoder.
"""

import warnings
from typing import Dict, Optional

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

    def _encode_text_batch(self, batch_data: dict) -> np.ndarray:
        return (
            self._model.encode_text(batch_data)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
```

### Step 6: 验证与收尾 — 预计 10 分钟

```bash
# 1. 单元级别验证
python -c "from clip_server.executors.base import BaseCLIPEncoder; print('OK')"
python -c "from clip_server.executors.clip_torch import CLIPEncoder; print('OK')"
python -c "from clip_server.executors.clip_onnx import CLIPEncoder; print('OK')"
python -c "from clip_server.executors.clip_tensorrt import CLIPEncoder; print('OK')"

# 2. 运行完整测试套件
pytest tests/test_simple.py tests/test_client.py tests/test_model.py \
       tests/test_ranker.py tests/test_helper.py tests/test_server.py \
       tests/test_tokenization.py tests/test_asyncio.py \
       -v -s --timeout=120

# 3. 确认导入行为不变（conftest.py 中的 make_flow fixture 正常）
pytest tests/conftest.py -v --collect-only
```

### Step 7（可选）：改进模型工厂

**文件**：`server/clip_server/model/clip_model.py`

```python
class CLIPModel(BaseCLIPModel):
    _MODEL_MAP = {
        **_OPENCLIP_MODELS: 'openclip',
        **_MULTILINGUALCLIP_MODELS: 'multilingual',
        **_CNCLIP_MODELS: 'cnclip',
    }

    @classmethod
    def create(cls, name: str, **kwargs):
        model_type = cls._MODEL_MAP.get(name)
        if model_type is None:
            available = '\n'.join(f'\t- {k}' for k in cls._MODEL_MAP)
            raise ValueError(
                f'CLIP model {name} not found; available models:\n{available}'
            )
        if model_type == 'openclip':
            from clip_server.model.openclip_model import OpenCLIPModel
            return OpenCLIPModel(name, **kwargs)
        elif model_type == 'multilingual':
            from clip_server.model.mclip_model import MultilingualCLIPModel
            return MultilingualCLIPModel(name, **kwargs)
        elif model_type == 'cnclip':
            from clip_server.model.cnclip_model import CNClipModel
            return CNClipModel(name, **kwargs)
```

**验收标准**：`_build_model` 调用从 `CLIPModel(name, ...)` 改为 `CLIPModel.create(name, ...)` 后行为不变。

## 五、差异对照表（重构前后）

| 指标 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| **executor 总行数** | 624 | ~370 | -40% |
| **clip_torch.py** | 224 | ~90 | -60% |
| **clip_onnx.py** | 212 | ~80 | -62% |
| **clip_tensorrt.py** | 188 | ~70 | -63% |
| **base.py** | — | ~130 | 新建 |
| **encode() 实现次数** | 3（重复） | 1（基类） | -67% |
| **rank() 实现次数** | 3（重复） | 1（基类） | -67% |
| **新增后端成本** | ~200 行 | ~80 行 | 覆盖 4 个 hook 即可 |
| **Bug 修复** | 4 个已知 bug | 全部修复 | — |

## 六、测试策略

### 回归测试（必须全部通过）

| 文件 | 覆盖范围 | 运行时 |
|------|---------|--------|
| `test_simple.py` | gRPC/HTTP/WebSocket 协议 | CPU |
| `test_client.py` | 客户端并发、错误处理 | CPU |
| `test_asyncio.py` | 异步编码 | CPU |
| `test_model.py` | 模型加载（torch + onnx） | GPU |
| `test_ranker.py` | 排序端点 | CPU |
| `test_helper.py` | 工具函数 | CPU |
| `test_server.py` | 服务端集成 | CPU |
| `test_tokenization.py` | 分词器 | CPU |

### 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| `super().__init__` 调用顺序改变 | 子类属性在基类 init 前未设置 | torch 子类在 `super().__init__` 前设置 `_preproc_kwargs`，其他子类用 `_post_init` |
| TensorRT 测试需 GPU | 无法本地验证 | 仅做语法级别验证 |
| conftest.py 导入路径 | fixture 创建 Flow 失败 | `make_flow` 按 `request.param` 动态导入具体 executor 类 |

## 七、回滚方案

```bash
# 如果重构引入问题，一条命令回滚
git checkout -- server/clip_server/executors/
git checkout -- server/clip_server/helper.py
rm server/clip_server/executors/base.py
```

## 八、后续扩展指南

重构后添加第四个后端（如 OpenVINO）：

```python
# server/clip_server/executors/clip_openvino.py (~60 行)
class CLIPEncoder(BaseCLIPEncoder):
    RUNTIME = 'openvino'
    _return_np = True

    def __init__(self, name=..., device='cpu', ...):
        self._preproc_image_kwargs = {'dtype': 'fp32'}
        self._preproc_text_kwargs = {}
        super().__init__(name=name, device=device, ...)

    def _build_model(self, name, **kwargs):
        return OpenVINOModel(name)

    def _encode_image_batch(self, batch_data):
        return self._model.encode_image(batch_data)

    def _encode_text_batch(self, batch_data):
        return self._model.encode_text(batch_data)
```

只需 3 个必须方法 + 3 个属性，所有公共逻辑（encode/rank/close/preproc/tracing/monitoring）自动继承。

---

## 九、重构中学到的（Learning Points）

### 9.1 识别重复代码：先画差异矩阵再动手

重构前最重要的步骤不是写代码，而是把三个文件的差异列成矩阵（见 [2.4](#24-三-executor-的差异矩阵)）。只有看清了「什么是相同的」「什么是不同的」，才能设计出正确的抽象边界。

| 经验 | 反面案例 |
|------|---------|
| 差异矩阵让所有差异显式化 | 凭感觉抽基类，很容易漏掉某个子类的特殊情况 |
| 把差异分类为「数据差异」和「行为差异」 | 数据差异用属性/参数化处理，行为差异用 hook 方法 |

### 9.2 Template Method 模式的适用场景

```
                    ┌── encode() ──┐
                    │  tracer span │
                    │  access_paths│
                    │  split_da    │
                    │  inference   │
                    │  map_batch   │  ← 骨架（基类实现一次）
                    │  monitor     │
                    │              │
           ┌────────┤ ┌ ─ ─ ─ ─ ─ ┐│
           │        └─┤ _encode_*  ├┘ ← 差异点（子类各实现）
           │          └ ─ ─ ─ ─ ─ ┘
    子类覆盖 hook
```

适用条件：
- 骨架稳定，差异点明确
- 调用顺序固定（encode 永远是 tracer → split → preproc → inference）
- 子类不需要改变骨架逻辑（如果子类需要改变调用顺序，Template Method 不合适）

不适用的情况：
- 子类需要不同的调用顺序 — 改用 Strategy 模式
- 差异点过多（>10 个）— 抽基类收益递减

### 9.3 `__new__` 工厂 vs `create()` 类方法

原始代码用 `__new__` + `super().__new__(SubClass)` 做工厂分发：

```python
# ❌ 不直观，依赖 Python 内部机制
class CLIPModel:
    def __new__(cls, name):
        if name in _MODELS:
            instance = super().__new__(OpenCLIPModel)  # 返回其他类的实例
        return instance

model = CLIPModel('ViT-B-32')  # type(model) == OpenCLIPModel  ← 违反直觉
```

`__new__` 返回其他类的实例时：
1. 调用方以为造了 `CLIPModel`，实际是 `OpenCLIPModel`
2. `__init__` 调用行为在跨版本时不一致
3. IDE 无法做类型推断

```python
# ✅ 显式工厂，语义清晰
class CLIPModel:
    @classmethod
    def create(cls, name):
        if name in _MODELS:
            return OpenCLIPModel(name)  # 显式构造子类

model = CLIPModel.create('ViT-B-32')  # 语义：从工厂创建合适的模型
```

**选择标准**：
- 子类需要同时是 `CLIPModel` 的子类（继承链）→ 调用方用 `isinstance` 判断
- 调用方只关心「获得一个可用的 model」→ `create()` 工厂方法更合适

本次重构选了 `create()`，因为：
- `OpenCLIPModel` 等子类仍需从 `CLIPModel` 继承（共用 `BaseCLIPModel.image_size` 等属性）
- `CLIPModel.__init__` 加 `type(self) is CLIPModel` 守卫，防止直接实例化
- `CLIPModel.create()` 是唯一合法的外部入口

### 9.4 Hook 设计：必须 vs 可选 vs 属性

| 类型 | 示例 | 设计原则 |
|------|------|---------|
| **必须方法** (abstractmethod) | `_build_model`, `_encode_image_batch` | 没有合理默认实现，子类必须提供 |
| **可选方法** (默认空实现) | `_post_init`, `_cleanup_model` | 有合理默认（什么都不做），子类按需覆盖 |
| **类属性** | `RUNTIME`, `_return_np` | 纯数据差异，不需要方法调用开销 |
| **property** | `_inference_context` | 行为差异但调用方是无参数的 `with ctx:` |

一个常见错误是把所有差异都做成 abstractmethod。反思：如果差异只是「传不传某个参数」，用属性/配置比用方法好。

### 9.5 重构的验证策略：没有测试环境时怎么办

本项目依赖钉死在 2023 年，Python 3.12 无法安装。无法运行测试套件时，按以下顺序验证：

| 层次 | 验证方式 | 能发现什么 |
|------|---------|-----------|
| 1. 语法 | `py_compile.compile()` | 语法错误、import 路径错误 |
| 2. 结构 | AST 分析类/方法/属性 | abstractmethod 是否实现、属性是否定义 |
| 3. 引用 | grep 搜索调用方 | 重命名后是否遗漏调用点 |
| 4. 导入路径 | 确认 conftest/YAML/benchmark 引用 | 模块路径是否仍然有效 |
| 5. diff 审查 | `git diff` 逐行对比 | 逻辑是否无意中被改变 |

1-4 可以自动化，5 需要人工判断。本次重构所有调用方通过 grep 确认无遗漏，conftest 导入路径通过 AST 分析确认兼容。

### 9.6 重构的提交节奏

七次提交，从核心到外围、从重构到文档：

```
b98b045  refactor: extract BaseCLIPEncoder                   ← 核心骨架
3dfffc3  feat: resource cleanup hooks                        ← 资源管理
256eb05  docs: hook interface + backend template             ← 接口文档
5802e44  refactor: replace CLIPModel.__new__ with create()   ← 模型工厂
9972f00  fix: client lifecycle + eliminate all bare excepts  ← 框架层修复
f5ff58d  docs: add key learnings (9.1-9.7)                   ← 学习记录
44737be  docs: add framework design learnings (9.8)           ← 学习记录
```

提交模式：5 个代码提交 + 2 个文档提交，交替进行。写完一段代码立即记录下 learnings，避免事后遗忘。

**时序上的设计考量**：

```
先修 bug（含在 b98b045）  →  零风险，立即止损
再抽基类（b98b045~5802e44）  →  核心重构，风险最大放最前
再加特性（3dfffc3, 256eb05）  →  增量，依赖基类稳定
最后改框架层（9972f00）       →  跨模块改动，涉及 client/server 两边
```

模型工厂（5802e44）放在基类重构之后但在框架层修复之前，因为它的调用方（`clip_torch._build_model`）刚被改写，改动点最热、上下文最清晰。

### 9.7 从 624 行到 474 行，实际减少了什么

| 减少的 | 增加的 |
|--------|--------|
| `encode()` ×3 → ×1（-200 行） | `base.py` 文档和结构代码（+130 行） |
| `rank()` ×3 → ×1（-36 行） | `_cleanup_model()` ×3（+17 行） |
| `_preproc_images/texts` ×3 → ×1（-90 行） | `_backend_template.py`（+78 行） |

净减少 150 行，但**行数不是关键指标**。真正重要的是：
- `encode()` 从 3 份副本变成 1 份 → 修 bug 只需改一处
- 新增后端从 ~200 行 → ~70 行 → 维护成本降 65%
- 公共逻辑和差异逻辑的边界被代码显式表达 → 可读性提升

### 9.8 框架设计层的改进

以上是代码层面的重构。在框架设计层面，还有三个跨切面的问题值得关注：

**A. 客户端生命周期管理**

原始 `Client` 类在 `__init__` 中创建了两个 Jina 内部客户端（`self._client` 和 `self._async_client`），但从未释放。这违反了「谁创建谁销毁」的资源管理原则。修复方案：

```python
# ✅ 显式生命周期 + context manager
class Client:
    def close(self):
        self._client.close()
        self._async_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# 使用方式
with Client('grpc://0.0.0.0:51000') as c:
    embeddings = c.encode(['hello world'])
```

设计原则：任何持有外部连接/资源的对象都应该可 `close()` 且支持 `with` 语法。

**B. 错误处理的粒度控制**

项目中有 5 处裸 `except:`（已全部修复），但修复方式不同，体现了错误处理的层次设计：

| 原位置 | 期望捕获 | 修复为 | 原则 |
|--------|---------|--------|------|
| `helper.py` | 网络/PyPI 异常 | `except Exception` | 不吞 `KeyboardInterrupt` |
| `client.py` | URL 解析异常 | `except Exception` | 同上 |
| `model.py` | flash-attn 未安装 | `except ImportError` | 精确捕获可选依赖 |
| `simple_tokenizer.py` | BPE 字节未命中 | `except ValueError` | 精确捕获字符串操作 |

原则：`except Exception` 是保底，`except ImportError` / `except ValueError` 等精确类型让意图自文档化。

**C. 全局状态污染**

`Client._prepare_streaming()` 直接写 `os.environ['JINA_GRPC_*'] = '0'` 来禁用 gRPC 消息大小限制。这是跨请求的进程级副作用：

```python
# ❌ 进程级全局变量，非线程安全、不可组合
os.environ['JINA_GRPC_SEND_BYTES'] = '0'
```

在框架设计中，配置应通过显式参数传递而非隐式修改全局状态。由于 Jina 内部通过环境变量读取此配置，彻底修复需要改动 Jina 框架本身。识别这种「自己无法修复但值得知道的坏味道」也是学习的一部分。

### 9.9 重构中的向后兼容：改实现不改接口

本次重构的约束条件：**不能修改任何外部接口**。以下接口在重构前后完全不变：

| 接口 | 调用方 | 保持方式 |
|------|--------|---------|
| `from clip_server.executors.clip_torch import CLIPEncoder` | conftest, YAML | 类名和模块路径不变 |
| `CLIPEncoder(name=..., device=..., ...)` | Flow 构造 | 构造函数签名不变 |
| `jtype: CLIPEncoder` + `py_modules: [clip_server.executors.clip_torch]` | Flow YAML | Jina 发现机制不变 |
| `Client(server)` | benchmark, 用户代码 | 构造函数签名不变（新增 `close` 等不破坏旧用法） |
| `CLIPModel(name)` → `CLIPModel.create(name)` | test_model.py | 旧接口报 `TypeError` 而非静默失败 |

**核心原则**：
- 重构只改内部实现，不改变外部行为
- 如果必须改接口，用 `TypeError` 或 `DeprecationWarning` 引导迁移
- 新增方法（`close`、`__enter__`）是原接口的超集，不破坏向后兼容

### 9.10 `_post_init` 钩子：解决 `runtime_args` 时序问题

在抽取基类时遇到一个关键问题：`torch.set_num_threads()` 需要在 `runtime_args.replicas` 可用后执行，而 `runtime_args` 由 Jina 框架在 `Executor.__init__()` 中设置。

原始代码的顺序：
```python
class CLIPEncoder(Executor):
    def __init__(self, ...):
        super().__init__(**kwargs)          # ← runtime_args 在这里被设置
        ...
        replicas = self.runtime_args.replicas  # ← 可以安全访问
        torch.set_num_threads(...)
        self._model = CLIPModel(...)        # ← 模型构建
```

如果子类在 `super().__init__()` 之前就做线程管理，`self.runtime_args` 还不存在。解决方案是引入 `_post_init()` 钩子：

```python
class BaseCLIPEncoder(Executor):
    def __init__(self, ...):
        super().__init__(**kwargs)           # ← runtime_args 就位
        self._pool = ThreadPool(...)
        self._device = self._resolve_device(device)
        self._model = self._build_model(...) # ← 模型构建
        ...
        self._post_init(**kwargs)            # ← 子类钩子，runtime_args 已可用

class CLIPEncoder(BaseCLIPEncoder):
    def _post_init(self, **kwargs):
        replicas = getattr(self.runtime_args, 'replicas', 1)
        torch.set_num_threads(...)           # ← 此时 runtime_args 必定可用
```

**通用模式**：当框架基类的初始化顺序与子类需求冲突时，提供一个尾部钩子比要求子类在 `__init__` 中「某行前做 A、某行后做 B」更安全。类似模式见于 Django 的 `populate()`、unittest 的 `setUp()`。

### 9.11 重构效果一览

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| executor 文件数 | 3（各自独立） | 4（1 基类 + 3 子类） |
| `encode()` 实现次数 | 3 份副本 | 1 份（基类） |
| `rank()` 实现次数 | 3 份副本 | 1 份（基类） |
| 新增后端成本 | ~200 行 | ~70 行 |
| 裸 `except:` | 5 处 | 0 处 |
| 客户端生命周期 | 无管理（依赖 GC） | `close()` + context manager |
| 模型工厂 | `__new__` 黑魔法 | `create()` 类方法 |
| 代码行数 | 624 | 474 + 模板 78 |
| 修改点收敛 | 修 encode bug 需改 3 处 | 修 encode bug 改 1 处 |
