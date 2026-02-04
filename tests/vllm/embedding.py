import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingCompletionRequest,
)

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
model_path = "/home/dev/model/Qwen/Qwen3-Embedding-0___6B/"

model = "qwem3-embedding"


async def main():
    # 1. 创建引擎
    engine_args = AsyncEngineArgs(
        model=model_path,
        runner="auto",
        convert="auto",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # model_config = ModelConfig()
    # 2. 创建模型管理器
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=model, model_path=model_path)],
        lora_modules=None,
    )

    # 3. 创建 OpenAIServingEmbedding 实例
    serving_embedding = OpenAIServingEmbedding(
        engine_client=engine,
        models=models,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        log_error_stack=False,
    )

    # 4. 创建 embedding 请求
    request = EmbeddingCompletionRequest(
        model=model,
        input=["我喜欢你", "我恨你"],
        encoding_format="float",
    )

    # 5. 调用 create_embedding 方法
    response = await serving_embedding.create_embedding(
        request=request,
        raw_request=None,
    )
    embeddings = []
    for i in response.data:
        embeddings.append(i.embedding)
    embeddings_np = np.array(embeddings)
    # u = np.array(embedding[0])  # “我喜欢你”
    # v = np.array(embedding[1])  # “我恨你”
    u = embeddings_np[0]
    v = embeddings_np[1]
    cos_sim = float(np.dot(u, v))  # 因为已经是单位向量
    print("余弦相似度：", cos_sim)


if __name__ == "__main__":
    asyncio.run(main())
