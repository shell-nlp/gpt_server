import httpx
from loguru import logger
from fastapi import HTTPException
import base64
import io
import os
from PIL import Image
import re
import torch
from transformers import AutoConfig
from transformers import AutoModel
import sentence_transformers


def is_base64_image(data_string):
    pattern = r"^data:image\/[a-zA-Z+]+;base64,[A-Za-z0-9+/]+=*$"
    return bool(re.match(pattern, data_string))


# 转换为Base64
def pil_to_base64(pil_img: Image.Image, format: str = "PNG"):
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)  # 明确指定PNG格式
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _extract_base64(data_url: str):
    """从Data URL中提取纯Base64数据"""
    return data_url.split(",", 1)[-1]  # 从第一个逗号后分割


async def _get_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载数据")
        return response.content


def bytesio2image(bytes_io: io.BytesIO) -> Image.Image:
    return Image.open(bytes_io)


def bytes2image(bytes_: bytes) -> Image.Image:
    bytes_io = io.BytesIO(bytes_)
    return Image.open(bytes_io)


async def load_base64_or_url(base64_or_url) -> io.BytesIO:
    # 根据 reference_audio 内容判断读取方式
    if base64_or_url.startswith("http://") or base64_or_url.startswith("https://"):
        audio_bytes = await _get_bytes_from_url(base64_or_url)
    else:
        try:
            if "data:" in base64_or_url:
                base64_or_url = _extract_base64(data_url=base64_or_url)
            audio_bytes = base64.b64decode(base64_or_url)
        except Exception as e:
            logger.warning("无效的 base64 数据: " + str(e))
            raise HTTPException(status_code=400, detail="无效的 base64 数据: " + str(e))
    # 利用 BytesIO 包装字节数据
    try:
        bytes_io = io.BytesIO(audio_bytes)
    except Exception as e:
        logger.warning("读取数据失败: " + str(e))
        raise HTTPException(status_code=400, detail="读取数据失败: " + str(e))
    return bytes_io


class PoolingModel:
    def __init__(self, model_path: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        architectures = getattr(model_config, "architectures", [])
        self.model = None
        self._pooling = None
        if "JinaForRanking" in architectures:
            self.model = AutoModel.from_pretrained(
                model_path,
                dtype="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            self.model.to(device)  # Move model to device

            def pooling_(self, query: str, documents: list):
                results = self.model.rerank(query, documents)
                embedding = [[i["relevance_score"]] for i in results]
                ret = {}
                ret["embedding"] = embedding
                ret["token_num"] = 0
                return ret

            self._pooling = pooling_
        elif "JinaVLForRanking" in architectures:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",
            )
            self.model.to(device)
            self.model.eval()
            logger.warning("model_type: JinaVLForRanking")

            def pooling_(self, query: str, documents: list):
                texts = documents
                sentence_pairs = [[query, inp] for inp in texts]
                query_type = doc_type = "text"

                if (
                    query.startswith("http://")
                    or query.startswith("https://")
                    or is_base64_image(query)
                ):
                    query_type = "image"
                if (
                    texts
                    and texts[0]
                    and (
                        texts[0].startswith("http://")
                        or texts[0].startswith("https://")
                        or is_base64_image(texts[0])
                    )
                ):
                    doc_type = "image"
                scores = self.model.compute_score(
                    sentence_pairs,
                    max_length=1024 * 2,
                    query_type=query_type,
                    doc_type=doc_type,
                )
                if isinstance(scores, float):
                    scores = [scores]
                embedding = [[float(score)] for score in scores]
                ret = {}
                ret["embedding"] = embedding
                ret["token_num"] = 0
                return ret

            self._pooling = pooling_
        else:
            mode = get_embedding_mode(model_path=model_path)
            if "embedding" == mode:
                self.model = sentence_transformers.SentenceTransformer(model_path)
                logger.warning("正在使用 embedding 模型...")
                encode_kwargs = {"normalize_embeddings": True, "batch_size": 64}

                def pooling_(self, query: str, documents: list = None):
                    texts = documents
                    outputs = self.model.tokenize(texts)
                    token_num = outputs["input_ids"].size(0) * outputs[
                        "input_ids"
                    ].size(1)
                    texts = list(map(lambda x: x.replace("\n", " "), texts))
                    embedding = self.model.encode(texts, **encode_kwargs).tolist()
                    ret = {}
                    ret["embedding"] = embedding
                    ret["token_num"] = token_num
                    return ret

                self._pooling = pooling_

            elif "rerank" == mode:
                self.model = sentence_transformers.CrossEncoder(model_name=model_path)
                logger.warning("正在使用 rerank 模型...")

                def pooling_(self, query: str, documents: list):
                    sentence_pairs = [[query, doc] for doc in documents]
                    scores = self.model.predict(sentence_pairs)
                    embedding = [[float(score)] for score in scores]
                    ret = {}
                    ret["embedding"] = embedding
                    ret["token_num"] = 0  # Rerank token num not typically calculated
                    return ret

                self._pooling = pooling_

            else:
                raise Exception(f"不支持的类型 mode: {mode}")

    def pooling(self, query, documents):
        if self._pooling is None:
            raise Exception("Model is not initialized or mode is not supported.")
        return self._pooling(self, query, documents)


def patch():
    class _HfFolder:
        pass

    import huggingface_hub

    huggingface_hub.HfFolder = _HfFolder
    logger.warning("patch huggingface_hub.HfFolder 成功！")


def get_embedding_mode(model_path: str):
    """获取模型的类型"""
    task_type = os.environ.get("task_type", "auto")
    if task_type == "embedding":
        return "embedding"
    elif task_type == "reranker":
        return "rerank"
    elif task_type == "classify":
        return "classify"

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type_text = getattr(
        getattr(model_config, "text_config", {}), "model_type", None
    )
    logger.warning(f"model_type: {model_type_text}")

    model_type = model_type_text
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
    r = get_embedding_mode("/home/dev/model/jinaai/jina-reranker-v3/")
    print(r)
