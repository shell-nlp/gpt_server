import httpx
from loguru import logger
from fastapi import HTTPException
import base64
import io
import os
from PIL.Image import Image
import re


def is_base64_image(data_string):
    pattern = r"^data:image\/[a-zA-Z+]+;base64,[A-Za-z0-9+/]+=*$"
    return bool(re.match(pattern, data_string))


# 转换为Base64
def pil_to_base64(pil_img: Image, format: str = "PNG"):
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)  # 明确指定PNG格式
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_base64(data_url: str):
    """从Data URL中提取纯Base64数据"""
    return data_url.split(",", 1)[-1]  # 从第一个逗号后分割


async def get_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载数据")
        return response.content


async def load_base64_or_url(base64_or_url):
    # 根据 reference_audio 内容判断读取方式
    if base64_or_url.startswith("http://") or base64_or_url.startswith("https://"):
        audio_bytes = await get_bytes_from_url(base64_or_url)
    else:
        try:
            if "data:" in base64_or_url:
                base64_or_url = extract_base64(data_url=base64_or_url)
            audio_bytes = base64.b64decode(base64_or_url)
        except Exception as e:
            logger.warning("无效的 base64 数据: " + str(e))
            raise HTTPException(status_code=400, detail="无效的 base64 数据: " + str(e))
    # 利用 BytesIO 包装字节数据，然后使用 soundfile 读取为 numpy 数组
    try:
        bytes_io = io.BytesIO(audio_bytes)
    except Exception as e:
        logger.warning("读取数据失败: " + str(e))
        raise HTTPException(status_code=400, detail="读取数据失败: " + str(e))
    return bytes_io


def get_embedding_mode(model_path: str):
    """获取模型的类型"""
    task_type = os.environ.get("task_type", "auto")
    if task_type == "embedding":
        return "embedding"
    elif task_type == "reranker":
        return "rerank"
    elif task_type == "classify":
        return "classify"
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = getattr(model_config, "architectures", [])
    model_type_text = getattr(
        getattr(model_config, "text_config", {}), "model_type", None
    )
    logger.warning(f"model_type: {model_type_text}")

    model_type = model_type_text
    # TODO --------- 在这里进行大过滤 ---------
    if "JinaVLForRanking" in architectures:
        logger.warning("model_type: JinaVLForRanking")
        return "vl_rerank"
    # --------- 在这里进行大过滤 ---------
    from infinity_emb import EngineArgs

    from infinity_emb.inference.select_model import get_engine_type_from_config

    engine_args = EngineArgs(
        model_name_or_path=model_path,
        engine="torch",
        embedding_dtype="float32",
        dtype="float32",
        bettertransformer=True,
    )
    engine_type = get_engine_type_from_config(engine_args)
    engine_type_str = str(engine_type)

    if "EmbedderEngine" in engine_type_str:
        return "embedding"
    elif "RerankEngine" in engine_type_str:
        return "rerank"
    elif "ImageEmbedEngine" in engine_type_str:
        return model_type or "image"
    elif "PredictEngine" in engine_type_str:
        return "classify"


if __name__ == "__main__":

    # 示例用法
    r = get_embedding_mode("/home/dev/model/jinaai/jina-reranker-m0/")
    print(r)
