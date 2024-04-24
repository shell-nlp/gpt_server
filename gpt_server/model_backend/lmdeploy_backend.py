import asyncio
from lmdeploy import (
    GenerationConfig,
    TurbomindEngineConfig,
    PytorchEngineConfig,
)

from lmdeploy.serve.async_engine import AsyncEngine


async def main():
    backend = "turbomind"
    if backend == "pytorch":
        backend_config = PytorchEngineConfig(model_name="", tp=1)
    if backend == "turbomind":
        backend_config = TurbomindEngineConfig(model_name="", tp=1)
    async_engine = AsyncEngine(
        model_path="/home/dev/model/qwen/Qwen-14B-Chat/",
        backend=backend,
        backend_config=backend_config,
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁"},
    ]
    gen_config = GenerationConfig(top_p=0.8, temperature=0.8, max_new_tokens=1024)
    results_generator = async_engine.generate(
        messages=messages, session_id=0, gen_config=gen_config
    )
    async for request_output in results_generator:
        print(request_output)


if __name__ == "__main__":
    asyncio.run(main())
