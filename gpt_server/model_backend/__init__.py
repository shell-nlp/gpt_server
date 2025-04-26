from lmdeploy.version import __version__

if __version__ == "0.7.3":
    from typing import Optional
    from lmdeploy.model import MODELS, InternLM2Chat7B

    @MODELS.register_module(name="internvl2_5", force=True)
    class InternVL2_5(InternLM2Chat7B):

        def __init__(
            self,
            meta_instruction="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",  # noqa
            **kwargs,
        ):
            super().__init__(meta_instruction=meta_instruction, **kwargs)

        @classmethod
        def match(cls, model_path: str) -> Optional[str]:
            """Return the model_name that was registered to MODELS.

            Args:
                model_path (str): the model path used for matching.
            """
            path = model_path.lower()
            if "internvl2.5" in path or "internvl2_5" in path or "internvl3" in path:
                return "internvl2_5"
