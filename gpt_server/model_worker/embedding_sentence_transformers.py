import os
from typing import List

from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_worker.utils import (
    PoolingModel,
)


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
        self.pool_model = PoolingModel(model_path=model_path)
        logger.warning(f"模型：{model_names[0]}")

    async def get_embeddings(self, params):
        self.call_ct += 1
        texts = params["input"]
        query = params.get("query", None)
        ret = self.pool_model.pooling(query=query, documents=texts)
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
