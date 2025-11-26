import os
from typing import List
from loguru import logger

from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_worker.utils import get_embedding_mode
import numpy as np
from vllm import LLM, EmbeddingRequestOutput, ScoringRequestOutput
from gpt_server.settings import get_model_config

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


def template_format(queries: List[str], documents: List[str]):
    model_config = get_model_config()
    hf_overrides = model_config.hf_overrides
    if hf_overrides:
        if hf_overrides["architectures"][0] == "Qwen3ForSequenceClassification":
            logger.info("使用 Qwen3ForSequenceClassification 模板格式化...")
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            instruction = "Given a web search query, retrieve relevant passages that answer the query"

            query_template = f"{prefix}<Instruct>: {instruction}\n<Query>: {{query}}\n"
            document_template = f"<Document>: {{doc}}{suffix}"
            queries = [query_template.format(query=query) for query in queries]
            documents = [document_template.format(doc=doc) for doc in documents]
            return queries, documents
    return queries, documents


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
        model_config = get_model_config()
        hf_overrides = model_config.hf_overrides
        self.mode = get_embedding_mode(model_path=model_path)
        runner = "auto"
        if self.model == "rerank":
            runner = "pooling"
        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=model_config.num_gpus,
            max_model_len=model_config.max_model_len,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            enable_prefix_caching=model_config.enable_prefix_caching,
            runner=runner,
            hf_overrides=hf_overrides,
        )

        logger.warning(f"模型：{model_names[0]}")
        logger.warning(f"正在使用 {self.mode} 模型...")

    async def get_embeddings(self, params):
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts: list = params["input"]
        embedding = []
        if self.mode == "embedding":
            texts = list(map(lambda x: x.replace("\n", " "), texts))
            # ----------
            outputs: list[EmbeddingRequestOutput] = self.engine.embed(
                texts,
                truncate_prompt_tokens=self.max_position_embeddings - 4,
            )
            embedding = [o.outputs.embedding for o in outputs]
            embeddings_np = np.array(embedding)
            # ------ L2归一化（沿axis=1，即对每一行进行归一化）-------
            norm = np.linalg.norm(embeddings_np, ord=2, axis=1, keepdims=True)
            normalized_embeddings_np = embeddings_np / norm
            embedding = normalized_embeddings_np.tolist()
        elif self.mode == "rerank":
            query = params.get("query", None)
            data_1 = [query] * len(texts)
            data_2 = texts
            data_1, data_2 = template_format(queries=data_1, documents=data_2)
            scores: list[ScoringRequestOutput] = self.engine.score(data_1, data_2)
            embedding = [[score.outputs.score] for score in scores]

        ret["embedding"] = embedding
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
