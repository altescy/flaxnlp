from typing import Any

from flaxnlp.models.model import Model


class TransformersModel:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, **kwargs: Any) -> Any:
        from transformers import FlaxAutoModel

        return FlaxAutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def from_config(model_type: str, **kwargs: Any) -> Any:
        from transformers import CONFIG_MAPPING, FlaxAutoModel

        return FlaxAutoModel.from_config(CONFIG_MAPPING[model_type](**kwargs))


Model.register("transformers::from_pretrained", constructor="from_pretrained")(TransformersModel)
Model.register("transformers::from_config", constructor="from_config")(TransformersModel)
