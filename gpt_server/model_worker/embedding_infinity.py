import os
from typing import List
import asyncio
from loguru import logger

from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
from infinity_emb.inference.select_model import get_engine_type_from_config
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase

label_to_category = {
    "S": "sexual",
    "H": "hate",
    "HR": "harassment",
    "SH": "self-harm",
    "S3": "sexual/minors",
    "H2": "hate/threatening",
    "V2": "violence/graphic",
    "V": "violence",
    "OK": "OK",
}


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
        logger.info(f"使用{device}加载...")
        model_type = getattr(self.model_config, "model_type", None)
        bettertransformer = True
        if model_type is not None and "deberta" in model_type:
            bettertransformer = False
        engine_args = EngineArgs(
            model_name_or_path=model_path,
            engine="torch",
            embedding_dtype="float32",
            dtype="float32",
            device=device,
            bettertransformer=bettertransformer,
        )
        engine_type = get_engine_type_from_config(engine_args)
        engine_type_str = str(engine_type)
        if "EmbedderEngine" in engine_type_str:
            self.mode = "embedding"
        elif "RerankEngine" in engine_type_str:
            self.mode = "rerank"
        elif "ImageEmbedEngine" in engine_type_str:
            self.mode = "image"
        self.engine: AsyncEmbeddingEngine = AsyncEngineArray.from_args([engine_args])[0]
        loop = asyncio.get_running_loop()
        loop.create_task(self.engine.astart())
        logger.info(f"正在使用 {self.mode} 模型...")
        logger.info(f"模型：{model_names[0]}")

    async def astart(self):
        await self.engine.astart()

    async def get_embeddings(self, params):
        logger.info(f"params {params}")
        logger.info(f"worker_id: {self.worker_id}")
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts: list = params["input"]
        if self.mode == "embedding":
            texts = list(map(lambda x: x.replace("\n", " "), texts))
            embeddings, usage = await self.engine.embed(sentences=texts)
            embedding = [embedding.tolist() for embedding in embeddings]
        elif self.mode == "rerank":
            query = params.get("query", None)
            ranking, usage = await self.engine.rerank(
                query=query, docs=texts, raw_scores=False
            )
            ranking = [
                {
                    "index": i.index,
                    "relevance_score": i.relevance_score,
                    "document": i.document,
                }
                for i in ranking
            ]
            ranking.sort(key=lambda x: x["index"])
            embedding = [
                [round(float(score["relevance_score"]), 6)] for score in ranking
            ]
        elif self.mode == "image":
            if (
                isinstance(texts[0], bytes)
                or "http" in texts[0]
                or "data:image" in texts[0]
            ):
                embeddings, usage = await self.engine.image_embed(images=texts)
            else:
                embeddings, usage = await self.engine.embed(sentences=texts)

            embedding = [embedding.tolist() for embedding in embeddings]
        ret["embedding"] = embedding
        ret["token_num"] = usage
        return ret

    async def classify(self, params):
        logger.info(f"params {params}")
        logger.info(f"worker_id: {self.worker_id}")
        self.call_ct += 1
        ret = {}
        texts = params["input"]
        scores, usage = await self.engine.classify(sentences=texts, raw_scores=False)
        results = []
        flagged = True
        for item in scores:
            categories_flags = {}
            category_scores = {}
            for entry in item:
                label = entry["label"]  # 原始的laebl
                label = label_to_category.get(
                    label, label
                )  # 将原始的label转换为category, 如果没有对应的category, 则使用原始的label
                score = entry["score"]
                # 更新类别标志和分数
                category_scores[label] = score
                # 如果分数高于某个阈值，标记为 flagged
                categories_flags[label] = False
                if score > 0.5:
                    categories_flags[label] = True
            results.append(
                {
                    "flagged": flagged,
                    "categories": categories_flags,
                    "category_scores": category_scores,
                }
            )
        ret["results"] = results
        ret["token_num"] = usage
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
