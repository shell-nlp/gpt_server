import os
from typing import List
import asyncio
from loguru import logger

from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
from infinity_emb.inference.select_model import get_engine_type_from_config
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_worker.utils import get_embedding_mode
import numpy as np
from vllm import LLM

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
        tensor_parallel_size = int(os.getenv("num_gpus", "1"))
        max_model_len = os.getenv("max_model_len", None)
        gpu_memory_utilization = float(os.getenv("gpu_memory_utilization", 0.8))
        enable_prefix_caching = bool(os.getenv("enable_prefix_caching", False))

        self.mode = get_embedding_mode(model_path=model_path)
        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
        )

        logger.warning(f"模型：{model_names[0]}")
        logger.warning(f"正在使用 {self.mode} 模型...")

    async def get_embeddings(self, params):
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts: list = params["input"]
        if self.mode == "embedding":
            texts = list(map(lambda x: x.replace("\n", " "), texts))
            # ----------
            outputs = self.engine.embed(texts)
            embedding = [o.outputs.embedding for o in outputs]
            embeddings_np = np.array(embedding)
            # ------ L2归一化（沿axis=1，即对每一行进行归一化）-------
            norm = np.linalg.norm(embeddings_np, ord=2, axis=1, keepdims=True)
            normalized_embeddings_np = embeddings_np / norm
            embedding = normalized_embeddings_np.tolist()

        ret["embedding"] = embedding
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
