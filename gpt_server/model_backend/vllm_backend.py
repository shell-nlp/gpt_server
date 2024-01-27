from typing import Any, Dict, AsyncGenerator
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from gpt_server.model_backend.base import ModelBackend


class VllmBackend(ModelBackend):
    def __init__(self, model_path) -> None:
        engine_args = AsyncEngineArgs(
            model_path, tensor_parallel_size=1, trust_remote_code=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def stream_chat(self, query: str, params: Dict[str, Any]) -> AsyncGenerator:
        request_id = params.get("request_id")
        top_p = params.get("top_p")
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")
        prompt_token_ids = params.get("prompt_token_ids")

        sampling = SamplingParams(
            use_beam_search=False,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        results_generator = self.engine.generate(
            query,
            sampling_params=sampling,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,  # 这个是不同之处
        )
        async for request_output in results_generator:
            yield request_output.outputs[0].text
