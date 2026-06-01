from clip_server.model.pretrained_models import (
    _OPENCLIP_MODELS,
    _MULTILINGUALCLIP_MODELS,
    _CNCLIP_MODELS,
    _VISUAL_MODEL_IMAGE_SIZE,
)


class BaseCLIPModel:
    def __init__(self, name: str, **kwargs):
        super().__init__()
        self._name = name

    @staticmethod
    def get_model_name(name: str):
        return name

    @property
    def model_name(self):
        return self.__class__.get_model_name(self._name)

    @property
    def image_size(self):
        return _VISUAL_MODEL_IMAGE_SIZE.get(self.model_name, None)


class CLIPModel(BaseCLIPModel):
    def __init__(self, name: str, **kwargs):
        if type(self) is CLIPModel:
            raise TypeError(
                'CLIPModel cannot be instantiated directly. '
                'Use CLIPModel.create(name) instead.'
            )
        super().__init__(name, **kwargs)

    @classmethod
    def create(cls, name: str, **kwargs):
        if name in _OPENCLIP_MODELS:
            from clip_server.model.openclip_model import OpenCLIPModel

            return OpenCLIPModel(name, **kwargs)
        elif name in _MULTILINGUALCLIP_MODELS:
            from clip_server.model.mclip_model import MultilingualCLIPModel

            return MultilingualCLIPModel(name, **kwargs)
        elif name in _CNCLIP_MODELS:
            from clip_server.model.cnclip_model import CNClipModel

            return CNClipModel(name, **kwargs)
        else:
            available = '\n'.join(
                f'\t- {k}'
                for k in sorted(
                    list(_OPENCLIP_MODELS.keys())
                    + list(_MULTILINGUALCLIP_MODELS.keys())
                    + list(_CNCLIP_MODELS.keys())
                )
            )
            raise ValueError(
                f'CLIP model {name} not found; available models:\n{available}'
            )
