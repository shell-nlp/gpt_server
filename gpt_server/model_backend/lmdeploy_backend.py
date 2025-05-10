import os
import sys
from lmdeploy import (
    GenerationConfig,
    TurbomindEngineConfig,
    PytorchEngineConfig,
)
from typing import Any, Dict, AsyncGenerator
from lmdeploy.archs import get_task
from gpt_server.model_handler.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.async_engine import get_names_from_model
from loguru import logger
from gpt_server.model_backend.base import ModelBackend

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
from lmdeploy.utils import get_logger

get_logger("lmdeploy").setLevel("WARNING")
os.environ["TM_LOG_LEVEL"] = "ERROR"


def is_stop(output: str, stop_str: str):
    # 直接创建子序列列表，不在每次调用中重复计算
    stop_str_sub_seq_list = [stop_str[: i + 1] for i in range(len(stop_str))]

    # 判断是否以 stop_str 子序列结尾
    for stop_str_sub_seq in stop_str_sub_seq_list:
        if output.endswith(stop_str_sub_seq):
            # 找到匹配的子序列，返回截断后的输出和状态
            sub_seq_len = len(stop_str_sub_seq)
            return output[:-sub_seq_len], stop_str_sub_seq == stop_str

    # 如果没有匹配的子序列且整体不在结尾，返回原始输出
    return output, False


def is_messages_with_tool(messages: list):
    flag = False
    for msg in messages:
        if "content" not in msg:
            flag = True
            break
    return flag


class LMDeployBackend(ModelBackend):
    def __init__(self, model_path) -> None:
        backend = backend_map[os.getenv("backend")]
        enable_prefix_caching = bool(os.getenv("enable_prefix_caching", False))
        max_model_len = os.getenv("max_model_len", None)
        gpu_memory_utilization = float(os.getenv("gpu_memory_utilization", 0.8))
        kv_cache_quant_policy = int(os.getenv("kv_cache_quant_policy", 0))
        dtype = os.getenv("dtype", "auto")
        logger.info(f"后端 {backend}")
        if backend == "pytorch":
            backend_config = PytorchEngineConfig(
                tp=int(os.getenv("num_gpus", "1")),
                dtype=dtype,
                session_len=int(max_model_len) if max_model_len else None,
                enable_prefix_caching=enable_prefix_caching,
                cache_max_entry_count=gpu_memory_utilization,
                quant_policy=kv_cache_quant_policy,
            )
        if backend == "turbomind":
            backend_config = TurbomindEngineConfig(
                tp=int(os.getenv("num_gpus", "1")),
                enable_prefix_caching=enable_prefix_caching,
                session_len=int(max_model_len) if max_model_len else None,
                dtype=dtype,
                cache_max_entry_count=gpu_memory_utilization,
                quant_policy=kv_cache_quant_policy,  # 默认为：0
            )
        pipeline_type, pipeline_class = get_task(model_path)
        logger.info(f"模型架构：{pipeline_type}")
        self.async_engine = pipeline_class(
            model_path=model_path,
            backend=backend,
            backend_config=backend_config,
        )
        model_type = get_names_from_model(model_path=model_path)[1]
        self.messages_type_select = (
            model_type[1] == "base"
        )  # 如果为True 则使用 prompt:str 否则： messages：list
        self.tokenizer = self.async_engine.tokenizer
        self.reasoning_parser_cache = {}

    async def stream_chat(self, params: Dict[str, Any]) -> AsyncGenerator:
        prompt = params.get("prompt", "")
        logger.info(prompt)
        messages = params["messages"]
        request_id = params.get("request_id", "0")
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        top_k = params.get("top_k", 50)

        max_new_tokens = min(int(params.get("max_new_tokens", 1024 * 8)), 1024 * 4)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_words_ids", None) or []
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        reasoning_parser_type = params.get("reasoning_parser", None)
        request = params.get("request", None)
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
        if params.get("tools", None) or is_messages_with_tool(messages=messages):
            messages = prompt or messages  # 解决lmdeploy 的提示模板不支持 tools
        if self.messages_type_select:
            messages = prompt or messages
        multimodal = params.get("multimodal", False)
        if multimodal:  # 多模态模型
            messages = params["messages"]
        results_generator = self.async_engine.generate(
            messages=messages, session_id=int(request_id), gen_config=gen_config
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
            # TODO ------------------------修复LMDeploy stop 无法停止的问题-------------------------------------------
            # output_info_list = []
            # for stop_str in list(stop):
            #     if stop_str:
            #         text, bool_value = is_stop(output=current_text, stop_str=stop_str)
            #         output_info_list.append(
            #             {"text": text, "bool_value": bool_value, "text_len": len(text)}
            #         )
            # output_info_list.sort(key=lambda x: x["text_len"])
            # output_info = output_info_list[0]
            # ret["text"] = output_info["text"][len(previous_text) :]
            # if output_info["bool_value"]:
            #     ret["finish_reason"] = "stop"
            #     yield ret
            #     break
            # TODO ------------------------修复LMDeploy stop 无法停止的问题-------------------------------------------
            if not ret["text"] and not ret.get("reasoning_content", ""):
                continue
            yield ret
            previous_text = current_text
        logger.info(current_text)
        logger.info(usage)
