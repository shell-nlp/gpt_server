import os
import sys
from lmdeploy import (
    GenerationConfig,
    TurbomindEngineConfig,
    PytorchEngineConfig,
)
from transformers import PreTrainedTokenizer
from typing import Any, Dict, AsyncGenerator, List, Optional
from lmdeploy.archs import get_task
from gpt_server.model_handler.reasoning_parser import ReasoningParserManager
from loguru import logger
from gpt_server.model_backend.base import ModelBackend
from gpt_server.settings import get_model_config
from lmdeploy.logger import RequestLogger
from lmdeploy.utils import get_logger

if sys.platform == "linux":
    # 防止Python c库没有加载导致lmdeploy pytorch后端报错
    os.environ["C_INCLUDE_PATH"] = "/usr/include/python3.8:" + (
        os.environ.get("C_INCLUDE_PATH", "")
    )
    os.environ["LUS_INCLUDE_PATH"] = "/usr/include/python3.8:" + (
        os.environ.get("LUS_INCLUDE_PATH", "")
    )
backend_map = {
    "lmdeploy-pytorch": "pytorch",  # pytorch后端
    "lmdeploy-turbomind": "turbomind",  # turbomind后端
}
# ------- 日志控制 -------
log_level = os.getenv("log_level", "WARNING")


get_logger("lmdeploy").setLevel(log_level)  # 默认WARNING
os.environ["TM_LOG_LEVEL"] = "WARNING"
# ------- 日志控制 -------


class CustomRequestLogger(RequestLogger):
    def log_prompt(self, session_id: int, prompt: str) -> None:
        if not isinstance(prompt, str):
            # Prompt may be a GPT4V message with base64 images;
            # logging might be impractical due to length
            return

    def log_inputs(
        self,
        session_id: int,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]],
        gen_config: GenerationConfig,
        adapter_name: str,
    ) -> None:
        max_log_len = self.max_log_len
        input_tokens = len(prompt_token_ids)
        if max_log_len is not None:
            if prompt is not None:
                prompt = prompt[:max_log_len]

            if prompt_token_ids is not None:
                prompt_token_ids = prompt_token_ids[:max_log_len]

        logger.info(
            f"session_id={session_id} adapter_name={adapter_name} gen_config={gen_config}"
        )
        logger.info(f"prompt：\n{prompt}")


class LMDeployBackend(ModelBackend):
    def __init__(self, model_path, tokenizer: PreTrainedTokenizer) -> None:
        model_config = get_model_config()
        logger.info(f"model_config: {model_config}")
        backend = backend_map[model_config.backend]
        logger.info(f"后端 {backend}")
        if backend == "pytorch":
            backend_config = PytorchEngineConfig(
                tp=model_config.num_gpus,
                dtype=model_config.dtype,
                session_len=model_config.max_model_len,
                enable_prefix_caching=model_config.enable_prefix_caching,
                cache_max_entry_count=model_config.gpu_memory_utilization,
                quant_policy=model_config.kv_cache_quant_policy,
            )
        if backend == "turbomind":
            backend_config = TurbomindEngineConfig(
                tp=model_config.num_gpus,
                enable_prefix_caching=model_config.enable_prefix_caching,
                session_len=model_config.max_model_len,
                dtype=model_config.dtype,
                cache_max_entry_count=model_config.gpu_memory_utilization,
                quant_policy=model_config.kv_cache_quant_policy,  # 默认为：0
            )
        pipeline_type, pipeline_class = get_task(model_path)
        logger.info(f"模型架构：{pipeline_type}")
        self.async_engine = pipeline_class(
            model_path=model_path,
            backend=backend,
            backend_config=backend_config,
        )
        self.tokenizer = self.async_engine.tokenizer
        self.reasoning_parser_cache = {}
        # 自定义日志
        self.async_engine.request_logger = CustomRequestLogger(max_log_len=None)

    def shutdown(self):
        logger.info("lmdeploy后端退出")

    async def stream_chat(self, params: Dict[str, Any]) -> AsyncGenerator:
        # params 已不需要传入 prompt
        messages = params["messages"]
        request_id = params.get("request_id", "0")
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        top_k = params.get("top_k", 50)
        max_new_tokens = int(params.get("max_new_tokens", 1024 * 8))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_words_ids", None) or []
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        reasoning_parser_type = params.get("reasoning_parser", None)
        request = params.get("request", None)
        enable_thinking = bool(params.get("enable_thinking", True))
        tools = params.get("tools", None)
        chat_template = params.get("chat_template", None)
        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)
        # prompt_token_ids = input_ids.tolist()[0]
        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        gen_config = GenerationConfig(
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,  # 存在问题
            top_k=50 if top_k == -1 else top_k,
            stop_words=list(stop),
            skip_special_tokens=True,
            response_format=params["response_format"],
        )

        results_generator = self.async_engine.generate(
            messages=messages,
            session_id=int(request_id),
            gen_config=gen_config,
            enable_thinking=enable_thinking,
            tools=tools,
            chat_template=chat_template,
        )
        usage = {}
        previous_text = ""
        current_text = ""
        previous_token_ids = []
        current_token_ids = []
        delta_token_ids = []
        async for request_output in results_generator:
            current_text = current_text + request_output.response

            usage = {
                "prompt_tokens": request_output.input_token_len,
                "completion_tokens": request_output.generate_token_len,
                "total_tokens": request_output.input_token_len
                + request_output.generate_token_len,
            }
            ret = {
                "text": request_output.response,
                "error_code": 0,
                "usage": usage,
                "finish_reason": request_output.finish_reason,
            }

            if reasoning_parser_type:
                reasoning_parser = None
                delta_token_ids = (
                    request_output.token_ids
                    if request_output.token_ids is not None
                    else []
                )
                current_token_ids = current_token_ids + delta_token_ids
                if reasoning_parser_type in self.reasoning_parser_cache:
                    reasoning_parser = self.reasoning_parser_cache.get(
                        reasoning_parser_type
                    )
                else:
                    reasoning_parser = ReasoningParserManager.get(
                        reasoning_parser_type
                    )(self.tokenizer)
                    self.reasoning_parser_cache[reasoning_parser_type] = (
                        reasoning_parser
                    )
                reasoning_delta = reasoning_parser.extract_reasoning_content_streaming(
                    previous_text=previous_text,
                    current_text=current_text,
                    delta_text=request_output.response,
                    previous_token_ids=previous_token_ids,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                )
                if reasoning_delta is not None:
                    ret["text"] = (
                        reasoning_delta.content if reasoning_delta.content else ""
                    )
                    ret["reasoning_content"] = (
                        reasoning_delta.reasoning_content
                        if reasoning_delta.reasoning_content
                        else ""
                    )
                previous_token_ids = current_token_ids

            if not ret["text"] and not ret.get("reasoning_content", ""):
                continue
            yield ret
            previous_text = current_text
        logger.info(current_text)
        logger.info(usage)
