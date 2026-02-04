import asyncio
import json
from typing import Any, Dict, AsyncGenerator
from gpt_server.model_backend.base import ModelBackend
from loguru import logger
from transformers import PreTrainedTokenizer
from gpt_server.settings import get_model_config
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    StreamOptions,
    ErrorResponse,
)
from sglang.srt.entrypoints.engine import (
    _launch_subprocesses,
    init_tokenizer_manager,
    run_scheduler_process,
    run_detokenizer_process,
)

from sglang.srt.server_args import ServerArgs
from starlette.responses import StreamingResponse


class SGLangBackend(ModelBackend):
    def __init__(self, model_path, tokenizer: PreTrainedTokenizer) -> None:
        model_config = get_model_config()
        self.lora_requests = []
        self.model_path = model_path
        # ---
        kwargs = {
            "model_path": model_path,
            "trust_remote_code": True,
            "mem_fraction_static": model_config.gpu_memory_utilization,
            "tp_size": model_config.num_gpus,
            "dtype": model_config.dtype,
            "context_length": model_config.max_model_len,
            "grammar_backend": "xgrammar",
            "disable_radix_cache": not model_config.enable_prefix_caching,
        }
        server_args = ServerArgs(**kwargs)

        tokenizer_manager, template_manager, scheduler_infos, port_args = (
            _launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=init_tokenizer_manager,
                run_scheduler_process_func=run_scheduler_process,
                run_detokenizer_process_func=run_detokenizer_process,
            )
        )
        self.tokenizer_manager = tokenizer_manager
        self.serving_chat = OpenAIServingChat(
            tokenizer_manager=tokenizer_manager, template_manager=template_manager
        )
        self.tokenizer = tokenizer

    def shutdown(self):
        logger.info("sglang后端退出")

    async def stream_chat(self, params: Dict[str, Any]) -> AsyncGenerator:
        # params 已不需要传入 prompt
        messages = params["messages"]
        tools = params.get("tools", None)
        chat_template = params.get("chat_template", None)
        enable_thinking = bool(params.get("enable_thinking", True))
        prompt = self.tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            enable_thinking=enable_thinking,
        )
        logger.info(f"prompt：\n{prompt}")
        request_id = params.get("request_id", "0")
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        top_k = params.get("top_k", -1)
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

        # ---- 支持 response_format ----
        response_format = params["response_format"]
        # ------
        if tools:
            for t in tools:
                if t["function"].get("strict", None) is None:
                    t["function"]["strict"] = False
        request = ChatCompletionRequest(
            messages=messages,
            model=self.model_path,
            max_tokens=max_new_tokens,
            temperature=temperature,
            seed=33,
            stream=True,
            stream_options=StreamOptions(
                include_usage=True, continuous_usage_stats=True
            ),
            tools=tools,
            response_format=response_format,
            stop_token_ids=stop_token_ids,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            top_k=top_k,
            top_p=top_p if top_p != 0 else 0.01,
            rid=request_id,
            # tool_choice=params.get("tool_choice", "auto"),
            chat_template_kwargs=None,
        )

        response = await self.serving_chat.handle_request(
            request=request, raw_request=None
        )
        try:
            if isinstance(response, StreamingResponse):
                output_text = ""
                pre_usage = None
                async for chunk in response.body_iterator:
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
                    if usage is None and pre_usage is not None:
                        usage = pre_usage
                    pre_usage = usage
                    try:
                        text = choices[0]["delta"]["content"]
                        if text is None:
                            text = ""
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
                logger.info(output_text)
                logger.info(usage)

            elif isinstance(response, ErrorResponse):
                pass

        except asyncio.CancelledError as e:
            self.tokenizer_manager.abort_request(request_id)
            logger.warning(f"request_id : {request_id} 已中断！")
