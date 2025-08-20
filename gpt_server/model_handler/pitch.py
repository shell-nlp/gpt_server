from typing import Optional
from flashtts.llm.vllm_generator import VllmGenerator
import flashtts
from loguru import logger


class VllmGenerator_(VllmGenerator):
    def __init__(
        self,
        model_path: str,
        max_length: int = 32768,
        gpu_memory_utilization: float = 0.6,
        device: str = "cuda",
        stop_tokens: Optional[list[str]] = None,
        stop_token_ids: Optional[list[int]] = None,
        **kwargs,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_kwargs = dict(
            model=model_path,
            max_model_len=max_length,
            gpu_memory_utilization=gpu_memory_utilization,
            # device=device,
            disable_log_stats=True,
            # disable_log_requests=True,
            **kwargs,
        )
        async_args = AsyncEngineArgs(**engine_kwargs)

        self.model = AsyncLLMEngine.from_engine_args(async_args)

        super(VllmGenerator, self).__init__(
            tokenizer=model_path,
            max_length=max_length,
            stop_tokens=stop_tokens,
            stop_token_ids=stop_token_ids,
        )


def pitch_flashtts():
    flashtts.llm.vllm_generator.VllmGenerator = VllmGenerator_
    logger.info("patch flashtts.llm.vllm_generator.VllmGenerator")
