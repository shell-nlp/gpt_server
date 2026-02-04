import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = "/home/dev/model/Qwen/Qwen3-30B-A3B-Instruct-2507/"

model = "qwem3vl"


class CustomOpenAIServingChat(OpenAIServingChat):
    async def render_chat_request(self, request):
        value = await super().render_chat_request(request)
        prompt = value[1][0]["prompt"]
        print("prompt:", prompt)
        return value


async def main():
    # 1. 创建引擎
    engine_args = AsyncEngineArgs(
        model=model_path,
        runner="auto",
        convert="auto",
        tensor_parallel_size=1,
        max_model_len=10240,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # model_config = ModelConfig()
    # 2. 创建模型管理器
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=model, model_path=model_path)],
        lora_modules=None,
    )

    # 3.
    serving_chat = CustomOpenAIServingChat(
        engine_client=engine,
        models=models,
        response_role="assistant",
        chat_template=None,
        chat_template_content_format="auto",
        request_logger=None,
    )

    # 4. 创建 embedding 请求
    request = ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "你是谁"}],
        max_tokens=100,
        temperature=1.0,
        seed=33,
        stream=True,
        stream_options=StreamOptions(include_usage=True, continuous_usage_stats=True),
    )

    # 5. 调用 create_chat 方法
    response = await serving_chat.create_chat_completion(
        request=request,
        raw_request=None,
    )
    async for chunk in response:
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
