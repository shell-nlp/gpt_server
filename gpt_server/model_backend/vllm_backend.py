import json
import os
from typing import Any, Dict, AsyncGenerator
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from fastchat.utils import is_partial_stop
from gpt_server.model_backend.base import ModelBackend
from loguru import logger
import vllm
from vllm.lora.request import LoRARequest
from vllm.entrypoints.chat_utils import (
    ConversationMessage,
    apply_hf_chat_template,
    load_chat_template,
    parse_chat_messages_futures,
)

# 解决vllm中 ray集群在 TP>1时死的Bug
import ray

ray.init(ignore_reinit_error=True, num_cpus=4)

vllm_version = vllm.__version__


class VllmBackend(ModelBackend):
    def __init__(self, model_path) -> None:
        lora = os.getenv("lora", None)
        tensor_parallel_size = int(os.getenv("num_gpus", "1"))
        max_loras = 1
        enable_lora = False
        self.lora_requests = []
        if lora:
            enable_lora = True
            lora_dict: dict = json.loads(lora)
            max_loras = len(lora_dict)
            for i, (lora_name, lora_path) in enumerate(lora_dict.items()):
                self.lora_requests.append(
                    LoRARequest(
                        lora_name=lora_name,
                        lora_int_id=i,
                        lora_local_path=lora_path,
                    )
                )

        engine_args = AsyncEngineArgs(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enable_chunked_prefill=False,
            enable_lora=enable_lora,
            max_loras=max_loras,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

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
        if input_ids is None:  # 多模态模型
            # ----------------------------------------------------------------
            tokenizer = await self.engine.get_tokenizer()
            model_config = await self.engine.get_model_config()
            conversation, mm_data_future = parse_chat_messages_futures(
                messages, model_config, tokenizer
            )
            prompt = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
                chat_template=tokenizer.get_chat_template(),
                add_generation_prompt=True,
            )
            mm_data = await mm_data_future
            inputs = {"multi_modal_data": mm_data, "prompt": prompt}
        else:
            prompt_token_ids = input_ids.tolist()[0]
            inputs = {"prompt": prompt, "prompt_token_ids": prompt_token_ids}
        # ----------------------------------------------------------------
        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
            temperature = 0.01

        sampling = SamplingParams(
            use_beam_search=False,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=list(stop),
            stop_token_ids=stop_token_ids,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        lora_request = None
        for lora in self.lora_requests:
            if params["model"] == lora.lora_name:
                lora_request = lora
                break

        results_generator = self.engine.generate(
            inputs=inputs,
            sampling_params=sampling,
            request_id=request_id,
            lora_request=lora_request,
        )

        async for request_output in results_generator:
            text_outputs = request_output.outputs[0].text
            partial_stop = any(is_partial_stop(text_outputs, i) for i in stop)
            # prevent yielding partial stop sequence
            if partial_stop:
                continue

            aborted = False
            if request and await request.is_disconnected():
                await self.engine.abort(request_id)
                request_output.finished = True
                aborted = True
                for output in request_output.outputs:
                    output.finish_reason = "abort"
            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            ret = {
                "text": text_outputs,
                "error_code": 0,
                "usage": usage,
                "finish_reason": request_output.outputs[0].finish_reason,
            }
            yield ret

            if aborted:
                break
        logger.info(f"Lora: {request_output.lora_request}")
        logger.info(text_outputs)
        logger.info(usage)
