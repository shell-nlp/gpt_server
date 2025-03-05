import multiprocessing
import os
from typing import Any, Dict, AsyncGenerator
from fastchat.utils import is_partial_stop
from gpt_server.model_backend.base import ModelBackend
from loguru import logger

import sglang as sgl


@sgl.function
def pipeline(s, prompt, max_tokens):
    for p in prompt:
        if isinstance(p, str):
            s += p
        else:
            s += sgl.image(p)
    s += sgl.gen("response", max_tokens=max_tokens)


class SGLangBackend(ModelBackend):
    def __init__(self, model_path) -> None:
        lora = os.getenv("lora", None)
        enable_prefix_caching = bool(os.getenv("enable_prefix_caching", False))
        max_model_len = os.getenv("max_model_len", None)
        tensor_parallel_size = int(os.getenv("num_gpus", "1"))
        gpu_memory_utilization = float(os.getenv("gpu_memory_utilization", 0.8))
        dtype = os.getenv("dtype", "auto")
        max_loras = 1
        enable_lora = False
        self.lora_requests = []
        # ---
        multiprocessing.set_start_method("spawn", force=True)
        runtime = sgl.Runtime(
            model_path=model_path,
            trust_remote_code=True,
            mem_fraction_static=gpu_memory_utilization,
            tp_size=tensor_parallel_size,
            dtype=dtype,
            context_length=int(max_model_len) if max_model_len else None,
            grammar_backend="xgrammar",
        )

        sgl.set_default_backend(runtime)

    async def stream_chat(self, params: Dict[str, Any]) -> AsyncGenerator:
        prompt = params.get("prompt", "")
        messages = params["messages"]
        logger.info(prompt)
        request_id = params.get("request_id", "0")
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        top_k = params.get("top_k", -1.0)
        max_new_tokens = int(params.get("max_new_tokens", 1024 * 8))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_words_ids", None) or []
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        request = params.get("request", None)
        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        input_ids = params.get("input_ids", None)
        text_outputs = ""
        state = pipeline.run(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            stop=stop,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_k=top_k,
            top_p=top_p,
            stream=True,
        )
        async for out, meta_info in state.text_async_iter(
            var_name="response", return_meta_data=True
        ):

            partial_stop = any(is_partial_stop(out, i) for i in stop)
            # prevent yielding partial stop sequence
            if partial_stop:
                continue
            text_outputs += out
            aborted = False
            prompt_tokens = meta_info["prompt_tokens"]
            completion_tokens = meta_info["completion_tokens"]
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            ret = {
                "text": text_outputs,
                "error_code": 0,
                "usage": usage,
                "finish_reason": meta_info["finish_reason"]["type"],
            }
            yield ret

            if aborted:
                break
        logger.info(text_outputs)
        logger.info(usage)
