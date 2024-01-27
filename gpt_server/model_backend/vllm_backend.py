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
        request_id = params.get("request_id", "0")
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        max_tokens = int(params.get("max_new_tokens", 512))
        input_ids = params.get("input_ids")
        prompt_token_ids = input_ids.tolist()[0]
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
