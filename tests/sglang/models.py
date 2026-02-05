import asyncio

import os
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
from starlette.responses import StreamingResponse
from sglang.srt.server_args import ServerArgs

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = "/home/dev/model/Qwen/Qwen2___5-VL-7B-Instruct/"

model = "qwem3vl"


class CustomOpenAIServingChat(OpenAIServingChat):
    def _process_messages(self, request, is_multimodal):
        value = super()._process_messages(request, is_multimodal)
        prompt = value.prompt
        print("prompt:\n" + prompt)
        return value


async def main():
    kwargs = {
        "model_path": model_path,
        "trust_remote_code": True,
        # "mem_fraction_static": model_config.gpu_memory_utilization,
        "tp_size": 1,
        # "dtype": model_config.dtype,
        # "context_length": model_config.max_model_len,
        # "grammar_backend": "xgrammar",
        # "disable_radix_cache": not model_config.enable_prefix_caching,
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

    serving_chat = CustomOpenAIServingChat(
        tokenizer_manager=tokenizer_manager, template_manager=template_manager
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "你是谁"}],
        model=model_path,
        max_tokens=100,
        temperature=1.0,
        seed=33,
        stream=True,
        stream_options=StreamOptions(include_usage=True, continuous_usage_stats=True),
        tools=None,
        response_format=None,
    )

    response = await serving_chat.handle_request(request=request, raw_request=None)
    if isinstance(response, StreamingResponse):
        async for chunk in response.body_iterator:
            print(chunk)
    elif isinstance(response, ErrorResponse):
        pass


if __name__ == "__main__":
    asyncio.run(main())
