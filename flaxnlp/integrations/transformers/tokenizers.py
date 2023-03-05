import contextlib

import colt

with contextlib.suppress(ModuleNotFoundError):
    from transformers import AutoTokenizer

    class TransformersTokenizer(colt.Registrable, AutoTokenizer):  # type: ignore[misc]
        pass

    TransformersTokenizer.register("transformers::from_pretrained", constructor="from_pretrained")(
        TransformersTokenizer
    )
