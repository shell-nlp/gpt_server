import json
from typing import Any, Dict, AsyncGenerator
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.sampling_params import StructuredOutputsParams
from gpt_server.model_backend.base import ModelBackend
from loguru import logger
from vllm.lora.request import LoRARequest
from transformers import PreTrainedTokenizer

from vllm.config.structured_outputs import StructuredOutputsConfig
from gpt_server.settings import get_model_config
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import StreamOptions
from dataclasses import dataclass, asdict


class CustomOpenAIServingChat(OpenAIServingChat):
    async def render_chat_request(self, request):
        value = await super().render_chat_request(request)
        try:
            prompt = value[1][0]["prompt"]
            logger.info("prompt:\n" + prompt)
        except Exception:
            logger.error("request:\n" + str(value))
        return value


class VllmBackend(ModelBackend):
    def __init__(self, model_path, tokenizer: PreTrainedTokenizer) -> None:
        self.model_path = model_path
        model_config = get_model_config()
        logger.info(f"model_config: {model_config}")
        max_loras = 1
        enable_lora = False
        self.lora_requests = []
        if model_config.lora:
            enable_lora = True
            lora_dict: dict = json.loads(model_config.lora)
            max_loras = len(lora_dict)
            for i, (lora_name, lora_path) in enumerate(lora_dict.items()):
                self.lora_requests.append(
                    LoRARequest(
                        lora_name=lora_name,
                        lora_int_id=i,
                        lora_local_path=lora_path,
                    )
                )
        from vllm.config.kv_transfer import KVTransferConfig

        self.engine_args = AsyncEngineArgs(
            model_path,
            tensor_parallel_size=model_config.num_gpus,
            trust_remote_code=True,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            enable_chunked_prefill=False,
            enable_lora=enable_lora,
            max_loras=max_loras,
            enable_prefix_caching=model_config.enable_prefix_caching,
            dtype=model_config.dtype,
            max_model_len=model_config.max_model_len,
            # guided_decoding_backend="xgrammar",
            # 支持LMCache的KV传输
            # kv_transfer_config=KVTransferConfig(
            #     kv_connector="LMCacheConnectorV1", kv_role="kv_both"
            # ),
            prefix_caching_hash_algo="xxhash",
            structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        )
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths=[
                BaseModelPath(name=self.model_path, model_path=self.model_path)
            ],
            lora_modules=None,
        )
        self.serving_chat = CustomOpenAIServingChat(
            engine_client=self.engine,
            models=models,
            response_role="assistant",
            chat_template=None,
            chat_template_content_format="auto",
            request_logger=None,
            trust_request_chat_template=True,
            enable_auto_tools=True,
            tool_parser=None,
        )

    def shutdown(self):
        self.engine.shutdown()
        logger.info("vllm后端退出")

    async def stream_chat(self, params: Dict[str, Any]) -> AsyncGenerator:
        # params 已不需要传入 prompt
        messages = params["messages"]
        request_id = params.get("request_id", "0")
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        top_k = int(params.get("top_k", 0))
        max_new_tokens = int(params.get("max_new_tokens", 1024 * 8))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_words_ids", None) or []
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        enable_thinking = bool(params.get("enable_thinking", True))
        request = params.get("request", None)
        tools = params.get("tools", None)
        chat_template = params.get("chat_template", None)
        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        # ----------------------------------------------------------------
        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
            temperature = 0.01
        response_format = params["response_format"]
        guided_json_object = None
        guided_decoding = None
        guided_json = None
        if response_format is not None:
            if response_format["type"] == "json_object":
                guided_json_object = True
            if response_format["type"] == "json_schema":
                json_schema = response_format["json_schema"]
                assert json_schema is not None
                guided_json = json_schema["schema"]
            guided_decoding = StructuredOutputsParams(
                json=guided_json,
                regex=None,
                choice=None,
                grammar=None,
                json_object=guided_json_object,
                whitespace_pattern=None,
            )
            if response_format["type"] == "text":
                guided_decoding = None

        lora_request = None
        for lora in self.lora_requests:
            if params["model"] == lora.lora_name:
                lora_request = lora
                break

        request = ChatCompletionRequest(
            model=self.model_path,
            messages=messages,
            seed=33,
            stream=True,
            stream_options=StreamOptions(
                include_usage=True, continuous_usage_stats=True
            ),
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            stop=stop,
            stop_token_ids=stop_token_ids,
            structured_outputs=asdict(guided_decoding) if guided_decoding else None,
            request_id=request_id,
            chat_template=chat_template,
            tools=tools,
        )
        response = await self.serving_chat.create_chat_completion(
            request=request,
            raw_request=None,
        )
        output_text = ""
        async for chunk in response:
            # data: {"id":"chatcmpl-bf6de7d56c9bfecc","object":"chat.completion.chunk","created":1769947499,"model":"qwem3vl","choices":[{"index":0,"delta":{"content":"你好","reasoning_content":null},"logprobs":null,"finish_reason":null,"token_ids":null}],"usage":{"prompt_tokens":10,"total_tokens":11,"completion_tokens":1}}
            # data: [DONE]
            chunk = chunk.strip("data: ").strip()
            if chunk == "[DONE]":
                break
            chunk_dict = json.loads(chunk)
            choices = chunk_dict["choices"]
            if not choices:
                continue
            usage = chunk_dict["usage"]
            try:
                text = choices[0]["delta"]["content"]
            except Exception:
                logger.error(
                    f"Error in processing chunk: {chunk_dict}",
                )
            output_text += text
            ret = {
                "text": text,
                "usage": usage,
                "error_code": 0,
                "finish_reason": choices[0]["finish_reason"],
                "reasoning_content": choices[0]["delta"]["reasoning_content"],
            }
            yield ret

        # logger.info(f"Lora: {request_output.lora_request}")
        logger.info(output_text)
        logger.info(usage)


if __name__ == "__main__":
    s = 'data: {"id":"chatcmpl-bf6de7d56c9bfecc","object":"chat.completion.chunk","created":1769947499,"model":"qwem3vl","choices":[{"index":0,"delta":{"content":"你好","reasoning_content":null},"logprobs":null,"finish_reason":null,"token_ids":null}],"usage":{"prompt_tokens":10,"total_tokens":11,"completion_tokens":1}}'
    v = s.strip("data: ").strip()
    import json

    print(json.loads(v))
