from fastchat.conversation import get_conv_template
from typing import Any, Dict, AsyncGenerator
import asyncio
import os

from gpt_server.model_backend.base import ModelBackend

os.system("clear")

conv = get_conv_template("chatglm3")
conv.append_message(conv.roles[0], "你是谁")
prompt = conv.get_prompt() + "<|assistant|>"

print(prompt)
print("---------------------------")

from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer

# 异步方法
model_path = "/home/dev/model/chatglm3-6b/"
engine_args = AsyncEngineArgs(
    model_path, tensor_parallel_size=1, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
input_ids = tokenizer.build_chat_input("你是谁", history=[], role="user")[
    "input_ids"
].tolist()[0]
print("完美", input_ids)  # 完美
print(tokenizer.decode(input_ids))
print("*************")
# inputs = tokenizer.encode(prompt, return_tensors="pt").tolist()[0]
inputs = tokenizer.encode_plus(prompt, return_tensors="pt", is_split_into_words=True)[
    "input_ids"
].tolist()[0]
print("探索", inputs)
print(tokenizer.decode(inputs))


class VllmBackend(ModelBackend):
    def __init__(self) -> None:
        model_path = "/home/dev/model/chatglm3-6b/"

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
