import os
from typing import List
import asyncio
from loguru import logger

from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase


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

        self.engine: AsyncEmbeddingEngine = AsyncEngineArray.from_args(
            [
                EngineArgs(
                    model_name_or_path=model_path,
                    engine="torch",
                    embedding_dtype="float32",
                    dtype="auto",
                    device=device,
                )
            ]
        )[0]
        loop = asyncio.get_running_loop()
        loop.create_task(self.engine.astart())
        self.mode = "embedding"
        # rerank
        for model_name in model_names:
            if "rerank" in model_name:
                self.mode = "rerank"
                break
        if self.mode == "rerank":
            logger.info("正在使用 rerank 模型...")
        elif self.mode == "embedding":
            logger.info("正在使用 embedding 模型...")

    async def astart(self):
        await self.engine.astart()
        
    def generate_stream_gate(self, params):
        pass

    async def get_embeddings(self, params):
        logger.info(f"params {params}")
        logger.info(f"worker_id: {self.worker_id}")
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts = params["input"]
        if self.mode == "embedding":
            texts = list(map(lambda x: x.replace("\n", " "), texts))
            embeddings, usage = await self.engine.embed(sentences=texts)
            embedding = [embedding.tolist() for embedding in embeddings]
        elif self.mode == "rerank":
            query = params.get("query", None)
            scores, usage = await self.engine.rerank(
                query=query, docs=texts, raw_scores=False
            )
            embedding = [[float(score)] for score in scores]
        ret["embedding"] = embedding
        ret["token_num"] = usage
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
