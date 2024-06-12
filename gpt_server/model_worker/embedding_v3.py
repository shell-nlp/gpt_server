from typing import List
from gpt_server.model_worker.base import ModelWorkerBase
import sentence_transformers
from infinity_emb.engine import AsyncEmbeddingEngine
from infinity_emb.args import EngineArgs
import asyncio
from loguru import logger


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
        # model_kwargs = {"device": "cuda"}
        self.encode_kwargs = {"normalize_embeddings": True, "batch_size": 64}
        self.mode = "embedding"
        # rerank
        for model_name in model_names:
            if "rerank" in model_name:
                self.mode = "rerank"
                break
        engine_args = EngineArgs(
            model_name_or_path=model_path,
            batch_size=64,
            compile=True,
        )
        self.engine = AsyncEmbeddingEngine.from_args(engine_args=engine_args)
        loop = asyncio.get_running_loop()
        loop.create_task(self.engine.astart())

    def generate_stream_gate(self, params):
        pass

    async def get_embeddings(self, params):
        print("params", params)
        print("worker_id:", self.worker_id)
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts = params["input"]
        if self.mode == "embedding":
            embedding, usage = await self.engine.embed(sentences=texts)
        elif self.mode == "rerank":
            query = params.get("query", None)
            ranking, usage = await self.engine.rerank(query=query, docs=texts)
            embedding = [[score] for score in ranking]
        ret["embedding"] = embedding
        ret["token_num"] = usage
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
