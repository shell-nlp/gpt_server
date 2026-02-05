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
from loguru import logger

os.environ["CUDA_VISIBLE_DEVICES"] = "1,6"
model_path = "/home/dev/model/Qwen/Qwen3-30B-A3B-Instruct-2507/"
model = "qwem3vl"


class CustomOpenAIServingChat(OpenAIServingChat):
    async def render_chat_request(self, request):
        value = await super().render_chat_request(request)
        try:
            prompt = value[1][0]["prompt"]
            logger.info("prompt:\n" + prompt)
        except Exception:
            logger.error("request:\n" + str(value))
        return value


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


async def main():
    # 1. 创建引擎
    engine_args = AsyncEngineArgs(
        model=model_path,
        runner="auto",
        convert="auto",
        tensor_parallel_size=2,
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
        enable_auto_tools=True,
        tool_parser="hermes",
    )

    # 4. 创建 embedding 请求
    request = ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "南京天气怎么样"}],
        max_tokens=100,
        temperature=1.0,
        seed=33,
        stream=True,
        stream_options=StreamOptions(include_usage=True, continuous_usage_stats=True),
        tools=tools,
        parallel_tool_calls=False,
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
