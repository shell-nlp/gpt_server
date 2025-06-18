import asyncio
import os
from typing import List

import sentence_transformers
import torch
from transformers import AutoConfig, AutoModel
from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_worker.utils import load_base64_or_url, get_embedding_mode


class EmbeddingWorker(ModelWorkerBase):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str = None,  # type: ignore
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
            model_type="embedding",
        )
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
            device = "cpu"
        else:
            device = "cuda"
        logger.warning(f"使用{device}加载...")
        model_kwargs = {"device": device}
        # TODO
        self.mode = get_embedding_mode(model_path=model_path)
        self.encode_kwargs = {"normalize_embeddings": True, "batch_size": 64}
        if "clip_text_model" in self.mode:  # clip text 模型
            self.client = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            if device == "cuda":
                self.client.to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                logger.info(f"device: {self.client.device}")
            self.client.set_processor(model_path)
            self.client.eval()
        elif "vl_rerank" == self.mode:
            self.client = AutoModel.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",
            )

            self.client.to("cuda")  # or 'cpu' if no GPU is available
            self.client.eval()
        elif "rerank" == self.mode:
            self.client = sentence_transformers.CrossEncoder(
                model_name=model_path, **model_kwargs
            )
            logger.warning("正在使用 rerank 模型...")
        elif "embedding" == self.mode:
            self.client = sentence_transformers.SentenceTransformer(
                model_path, **model_kwargs
            )
            logger.warning("正在使用 embedding 模型...")
        logger.warning(f"模型：{model_names[0]}")

    async def get_embeddings(self, params):
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts = params["input"]
        if self.mode == "embedding":
            outputs = self.client.tokenize(texts)
            token_num = outputs["input_ids"].size(0) * outputs["input_ids"].size(1)
            texts = list(map(lambda x: x.replace("\n", " "), texts))
            embedding = self.client.encode(texts, **self.encode_kwargs).tolist()
        elif self.mode == "rerank":
            query = params.get("query", None)
            # outputs = self.client.tokenizer.tokenize(texts)
            # token_num = len(outputs)
            # TODO 暂时不计算 rerank token num
            token_num = 0
            sentence_pairs = [[query, inp] for inp in texts]
            scores = self.client.predict(sentence_pairs)
            embedding = [[float(score)] for score in scores]
        elif self.mode == "vl_rerank":
            query = params.get("query", None)
            token_num = 0
            sentence_pairs = [[query, inp] for inp in texts]
            query_type = doc_type = "text"
            if (
                query.startswith("http://")
                or query.startswith("https://")
                or "data:" in query
            ):
                query_type = "image"
            if (
                texts[0].startswith("http://")
                or texts[0].startswith("https://")
                or "data:" in texts[0]
            ):
                doc_type = "image"
            scores = self.client.compute_score(
                sentence_pairs,
                max_length=1024 * 2,
                query_type=query_type,
                doc_type=doc_type,
            )
            if isinstance(scores, float):
                scores = [scores]
            embedding = [[float(score)] for score in scores]
        elif self.mode == "clip_text_model":
            token_num = 0
            if isinstance(texts[0], dict):
                text = [i["text"] for i in texts]
                text = list(map(lambda x: x.replace("\n", " "), text))

                images = [i["image"] for i in texts]
                coro_list = []
                for i in images:
                    coro = load_base64_or_url(base64_or_url=i)
                    coro_list.append(coro)
                result_images = await asyncio.gather(*coro_list)

                embedding = self.client.encode(
                    images=result_images,
                    text=text,
                ).tolist()
            elif isinstance(texts[0], str):
                if "http" in texts[0] or "data:image" in texts[0]:  # 图片
                    images = texts
                    coro_list = []
                    for i in images:
                        coro = load_base64_or_url(base64_or_url=i)
                        coro_list.append(coro)
                    result_images = await asyncio.gather(*coro_list)
                    embedding = self.client.encode(
                        images=result_images,
                    ).tolist()
                else:  # 文本
                    embedding = self.client.encode(
                        text=texts,
                    ).tolist()
        ret["embedding"] = embedding
        ret["token_num"] = token_num
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
