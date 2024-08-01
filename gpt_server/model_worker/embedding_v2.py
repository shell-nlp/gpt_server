import os
from typing import List
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
import sentence_transformers
import asyncio
from asyncio import Queue
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
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
            device = "cpu"
        else:
            device = "cuda"
        logger.info(f"使用{device}加载...")
        model_kwargs = {"device": device}
        self.request_queue: Queue = Queue()
        self.loop = asyncio.get_running_loop()

        self.worker_tasks = [
            self.loop.create_task(self.batch_processor()) for _ in range(1)
        ]
        # -------------------------------------------------------------------------
        self.batch_size = 64
        self.encode_kwargs = {
            "normalize_embeddings": True,
            "batch_size": self.batch_size,
        }
        self.mode = "embedding"
        # rerank
        for model_name in model_names:
            if "rerank" in model_name:
                self.mode = "rerank"
                break
        if self.mode == "rerank":
            self.client = sentence_transformers.CrossEncoder(
                model_name=model_path, **model_kwargs
            )
            logger.info("正在使用 rerank 模型...")
        elif self.mode == "embedding":
            self.client = sentence_transformers.SentenceTransformer(
                model_path, **model_kwargs
            )
            logger.info("正在使用 embedding 模型...")
        self.warm_up()

    def warm_up(self):
        logger.info("开始 warm_up")
        if self.mode == "embedding":
            self.client.encode(sentences=["你是谁"] * 10)
        elif self.mode == "rerank":
            self.client.predict(sentences=[["你好", "你好啊"]] * 10)

    def generate_stream_gate(self, params):
        pass

    async def batch_processor(self):
        logger.warning("进入batch_processor")
        while True:
            requests = []
            batch_size = 0
            try:
                while batch_size < self.batch_size:
                    # 等待 100ms
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=0.1
                    )
                    requests.append(request)
                    batch_size += len(request[0]["input"])

            except asyncio.TimeoutError as e:
                pass
            if requests:
                try:
                    all_input = [request[0]["input"] for request in requests]
                    futures = [request[1] for request in requests]

                    if self.mode == "embedding":
                        # 开始进行动态组批
                        ## 1. 展平text
                        # all_input = [ List[str] ]
                        # request[0] ---> params
                        all_texts = [text for input in all_input for text in input]
                        logger.debug(all_texts)
                        embeddings = self.client.encode(
                            all_texts, **self.encode_kwargs
                        ).tolist()

                    elif self.mode == "rerank":
                        # all_input = [ List[str] ]
                        # all_query = [str]
                        # all_texts = [str]
                        # request[0] ---> params
                        all_query = [request[0]["query"] for request in requests]
                        all_sentence_pairs = []

                        for query, inps in zip(all_query, all_input):
                            sentence_pairs = [[query, inp] for inp in inps]

                            all_sentence_pairs.extend(sentence_pairs)
                        logger.debug(all_sentence_pairs)
                        scores = self.client.predict(all_sentence_pairs)
                        embeddings = [[float(score)] for score in scores]

                    idx = 0
                    for future, request in zip(futures, requests):
                        num_texts = len(request[0]["input"])
                        future.set_result(embeddings[idx : idx + num_texts])
                        idx += num_texts
                except Exception as e:
                    logger.exception(e)
                    for future in futures:
                        future.set_exception(e)

    async def add_request(self, params: dict, future: asyncio.Future):

        await self.request_queue.put(item=(params, future))

    async def aembed(self, params: dict, future: asyncio.Future):
        await self.add_request(params, future)

    async def rerank(self, params: dict, future: asyncio.Future):
        await self.add_request(params, future)

    async def get_embeddings(self, params):
        logger.info(f"params {params}")
        logger.info(f"worker_id: {self.worker_id}")
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts = params["input"]
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if self.mode == "embedding":
            token_num = 0
            await self.aembed(params, future)
            embedding = await future
        elif self.mode == "rerank":
            token_num = 0
            await self.rerank(params, future)
            embedding = await future
        ret["embedding"] = embedding
        ret["token_num"] = token_num
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
    asyncio.run()
