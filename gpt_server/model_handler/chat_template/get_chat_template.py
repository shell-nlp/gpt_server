from pathlib import Path
from typing import Literal

cur_path = Path(__file__).parent


def get_chat_template(model_name: str = "", lang: Literal["en", "zh"] = "en") -> str:
    """获取chat_template

    Parameters
    ----------
    model_name : str
        模型名称
    lang : str, optional
        语言, by default en

    Returns
    -------
    str
        chat_template
    """
    suffix = ""
    if lang == "zh":
        suffix = "_zh"
    if model_name in ["qwen3", "qwen2_5", "qwen"]:
        with open(cur_path / f"qwen3{suffix}.jinja", "r", encoding="utf8") as f:
            return f.read()


if __name__ == "__main__":

    chat_template = get_chat_template("qwen3", lang="zh")
    print(chat_template)
