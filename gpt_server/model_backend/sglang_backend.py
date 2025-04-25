import base64
from io import BytesIO
import os
from typing import Any, Dict, AsyncGenerator, List, Optional
from fastchat.utils import is_partial_stop
from gpt_server.model_backend.base import ModelBackend
from loguru import logger
from PIL import Image
import sglang as sgl
from sglang.utils import convert_json_schema_to_str
from sglang.srt.conversation import generate_chat_conv

from qwen_vl_utils import process_vision_info


def _transform_messages(
    messages,
):
    transformed_messages = []
    for msg in messages:
        new_content = []
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            new_content.append({"type": "text", "text": content})
        elif isinstance(content, List):
            for item in content:  # type: ignore
                if "text" in item:
                    new_content.append({"type": "text", "text": item["text"]})
                elif "image_url" in item:
                    new_content.append(
                        {"type": "image", "image": item["image_url"]["url"]}
                    )
                elif "video_url" in item:
                    new_content.append(
                        {"type": "video", "video": item["video_url"]["url"]}
                    )
        new_message = {"role": role, "content": new_content}
        transformed_messages.append(new_message)

    return transformed_messages


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
        self.async_engine = sgl.Engine(
            model_path=model_path,
            trust_remote_code=True,
            mem_fraction_static=gpu_memory_utilization,
            tp_size=tensor_parallel_size,
            dtype=dtype,
            context_length=int(max_model_len) if max_model_len else None,
            grammar_backend="xgrammar",
            disable_radix_cache=not enable_prefix_caching,
        )

    async def stream_chat(self, params: Dict[str, Any]) -> AsyncGenerator:
        prompt = params.get("prompt", "")
        messages = params["messages"]
        logger.info(prompt)
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
        base64_images = []
        multimodal = params.get("multimodal", False)
        if multimodal:  # 多模态模型
            _messages = _transform_messages(messages)
            images, video_inputs = process_vision_info(_messages)
            if video_inputs:
                raise ValueError("Not support video input now.")
            if images:
                for image in images:
                    if isinstance(image, Image.Image):
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG", quality=100)
                        base64_images.append(
                            base64.b64encode(buffered.getvalue()).decode()
                        )
                    elif isinstance(image, str):
                        base64_images.append(image)
                    else:
                        raise ValueError(
                            f"Unsupported image type: {type(image)}, only support PIL.Image and base64 string"
                        )
        # ---- 支持 response_format ----
        response_format = params["response_format"]
        json_schema = None
        if response_format is not None:
            if response_format["type"] == "json_schema":
                json_schema = convert_json_schema_to_str(
                    response_format["json_schema"]["schema"]
                )
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "stop_token_ids": stop_token_ids,
            "stop": stop,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "top_k": top_k,
            "top_p": top_p if top_p != 0 else 0.01,
            "json_schema": json_schema,
        }
        generator = await self.async_engine.async_generate(
            prompt=prompt,
            sampling_params=sampling_params,
            stream=True,
            image_data=base64_images if base64_images else None,
        )
        previous_text = ""
        aborted = False
        async for chunk in generator:
            current_text = chunk["text"]
            meta_info = chunk["meta_info"]
            delta_text = current_text[len(previous_text) :]

            prompt_tokens = meta_info["prompt_tokens"]
            completion_tokens = meta_info["completion_tokens"]
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            ret = {
                "text": delta_text,
                "error_code": 0,
                "usage": usage,
                "finish_reason": (
                    meta_info["finish_reason"]["type"]
                    if meta_info["finish_reason"]
                    else None
                ),
            }
            if not ret["text"]:
                continue
            yield ret
            previous_text = current_text
            if aborted:
                break
        logger.info(current_text)
        logger.info(usage)
