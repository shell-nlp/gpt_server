"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)
- Moderations. (Reference: https://platform.openai.com/docs/api-reference/moderations)
- Audio. (Reference: https://platform.openai.com/docs/api-reference/audio)
"""

import asyncio
import argparse
import copy
from http import HTTPStatus
import json
import threading
import os
import time
import traceback
from typing import AsyncGenerator, Callable, Generator, Optional, Union, Dict, List, Any

import aiohttp
import fastapi
from fastapi import Depends, File, HTTPException, Request, responses, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import httpx
import base64

try:
    from pydantic.v1 import BaseSettings, validator
except ImportError:
    from pydantic import BaseSettings
import orjson
import shortuuid
import tiktoken
import uvicorn

from fastchat.constants import (
    WORKER_API_TIMEOUT,
    WORKER_API_EMBEDDING_BATCH_SIZE,
    ErrorCode,
)
from fastchat.protocol.openai_api_protocol import (
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    LogProbs,
    ModelList,
    ModelPermission,
)
from fastchat.protocol.api_protocol import (
    APITokenCheckRequest,
    APITokenCheckResponse,
    APITokenCheckResponseItem,
)
from loguru import logger

conv_template_map = {}

fetch_timeout = aiohttp.ClientTimeout(total=3 * 3600)


async def fetch_remote(url, pload=None, name=None):
    async with aiohttp.ClientSession(timeout=fetch_timeout) as session:
        async with session.post(url, json=pload) as response:
            chunks = []
            if response.status != 200:
                ret = {
                    "text": f"{response.reason}",
                    "error_code": ErrorCode.INTERNAL_ERROR,
                }
                return json.dumps(ret)

            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = b"".join(chunks)

    if name is not None:
        res = json.loads(output)
        if name != "":
            res = res[name]
        return res

    return output


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"
    api_keys: Optional[List[str]] = None

    @validator("api_keys", pre=True)
    def split_api_keys(cls, v):
        if isinstance(v, str):
            return v.split(",") if v else None
        return v

    class Config:
        # 关闭默认 JSON 解析行为
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            return raw_val  # 返回原始字符串，不解析成 JSON


app_settings = AppSettings()
from contextlib import asynccontextmanager

model_address_map = {}
models_ = []


async def timing_tasks():
    """定时任务"""
    global model_address_map, models_
    controller_address = app_settings.controller_address

    while True:
        try:
            # ret = await fetch_remote(controller_address + "/refresh_all_workers")
            models = await fetch_remote(
                controller_address + "/list_models", None, "models"
            )
            worker_addr_coro_list = []
            for model in models:
                worker_addr_coro = fetch_remote(
                    controller_address + "/get_worker_address",
                    {"model": model},
                    "address",
                )
                worker_addr_coro_list.append(worker_addr_coro)
            worker_address_list = await asyncio.gather(*worker_addr_coro_list)
            for model, worker_addr in zip(models, worker_address_list):
                model_address_map[model] = worker_addr
            models_ = list(model_address_map.keys())
            await asyncio.sleep(6)
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(6)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    logger.info(f"app_settings: {app_settings}")
    asyncio.create_task(timing_tasks())
    yield


app = fastapi.FastAPI(docs_url="/", lifespan=lifespan)
headers = {"User-Agent": "gpt_server API Server"}
get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


def check_model(model: str) -> Optional[JSONResponse]:
    global model_address_map, models_
    ret = None
    models = models_
    if model not in models_:
        ret = create_error_response(
            ErrorCode.INVALID_MODEL,
            f"Only {'&&'.join(models)} allowed now, your model {model}",
        )
    return ret


def process_input(model_name, inp):
    if isinstance(inp, str):
        inp = [inp]
    elif isinstance(inp, list):
        if isinstance(inp[0], int):
            try:
                decoding = tiktoken.model.encoding_for_model(model_name)
            except KeyError:
                logger.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                decoding = tiktoken.get_encoding(model)
            inp = [decoding.decode(inp)]
        elif isinstance(inp[0], list):
            try:
                decoding = tiktoken.model.encoding_for_model(model_name)
            except KeyError:
                logger.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                decoding = tiktoken.get_encoding(model)
            inp = [decoding.decode(text) for text in inp]

    return inp


def create_openai_logprobs(logprob_dict):
    """Create OpenAI-style logprobs."""
    return LogProbs(**logprob_dict) if logprob_dict is not None else None


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


def get_gen_params(
    model_name: str,
    worker_addr: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    presence_penalty: Optional[float],
    frequency_penalty: Optional[float],
    max_tokens: Optional[int],
    echo: Optional[bool],
    logprobs: Optional[int] = None,
    stop: Optional[Union[str, List[str]]],
    best_of: Optional[int] = None,
    use_beam_search: Optional[bool] = None,
    tools: Optional[list] = None,
    tool_choice=None,
    response_format=None,
    reasoning_parser: str = None,
    enable_thinking: bool = True,
) -> Dict[str, Any]:
    images = []
    if isinstance(messages, str):
        images = []

    prompt = ""
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "logprobs": logprobs,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "max_new_tokens": max_tokens,
        "echo": echo,
    }

    if len(images) > 0:
        gen_params["images"] = images

    if best_of is not None:
        gen_params.update({"best_of": best_of})
    if use_beam_search is not None:
        gen_params.update({"use_beam_search": use_beam_search})

    new_stop = set()
    _add_to_set(stop, new_stop)

    gen_params["stop"] = list(new_stop)
    # ------- TODO add messages tools -------
    gen_params["messages"] = messages
    gen_params["tools"] = tools
    gen_params["tool_choice"] = tool_choice
    # ------- TODO add messages tools -------
    gen_params["response_format"] = response_format
    gen_params["reasoning_parser"] = reasoning_parser
    gen_params["enable_thinking"] = enable_thinking
    return gen_params


class AddressManager:
    def __init__(self):

        self.lock = threading.Lock()
        self.last_index = -1  # 轮询索引

    def get_address(self, model):
        global model_address_map
        ips = model_address_map[model]
        self.worker_addr_list = ips.split(",")
        with self.lock:
            current_list = self.worker_addr_list.copy()

        if not current_list:
            return None

        n = len(current_list)
        if n == 1:
            return current_list[0]

        # 计算下一个索引（若列表长度变化，自动取模）
        self.last_index = (self.last_index + 1) % n
        return current_list[self.last_index]


address_manager = AddressManager()


def get_worker_address(model_name: str) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    # global model_address_map
    # worker_addr = model_address_map[model_name]
    worker_addr = address_manager.get_address(model=model_name)

    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")
    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


async def get_conv(model_name: str, worker_addr: str):
    conv_template = conv_template_map.get((worker_addr, model_name))
    if conv_template is None:
        conv_template = await fetch_remote(
            worker_addr + "/worker_get_conv_template", {"model": model_name}, "conv"
        )
        conv_template_map[(worker_addr, model_name)] = conv_template
    return conv_template


from gpt_server.openai_api_protocol.custom_api_protocol import CustomModelCard


@app.get(
    "/v1/models",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
async def show_available_models():
    controller_address = app_settings.controller_address
    ret = await fetch_remote(controller_address + "/refresh_all_workers")
    models = await fetch_remote(controller_address + "/list_models", None, "models")

    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(
            CustomModelCard(id=m, root=m, permission=[ModelPermission()])
        )
    return ModelList(data=model_cards)


from gpt_server.openai_api_protocol.custom_api_protocol import (
    CustomChatCompletionRequest,
    EmbeddingsResponse,
    CustomChatMessage,
    CustomChatCompletionResponse,
    CustomChatCompletionResponseChoice,
    CustomCompletionResponseChoice,
    ResponsesRequest,
    ErrorResponseV2,
    ErrorInfo,
    ResponsesResponse,
    ResponseOutputMessage,
    ResponseOutputText,
    UsageInfo,
)
from vllm.utils import random_uuid


@app.get(
    "/get_model_address_map",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
def get_model_address_map():
    global model_address_map
    return model_address_map


response_store_lock = asyncio.Lock()
response_store: dict[str, ResponsesResponse] = {}
msg_store = {}


@app.post(
    "/v1/responses",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
async def create_responses(request: ResponsesRequest):
    request.store = False  # 暂时关闭
    global response_store, msg_store
    error_check_ret = check_model(request.model)
    if error_check_ret is not None:
        return error_check_ret
    worker_addr = get_worker_address(request.model)
    max_tokens = 1024 * 8
    if request.max_output_tokens:
        max_tokens = request.max_output_tokens
    # ----------- 需要进行转化映射 -----------
    # Handle the previous response ID.
    previous_response_id = request.previous_response_id
    if previous_response_id is not None:
        async with response_store_lock:
            prev_response = response_store.get(previous_response_id)
        if prev_response is None:
            return ErrorResponseV2(
                error=ErrorInfo(
                    message=f"Response with id '{previous_response_id}' not found.",
                    type="invalid_request_error",
                    code=HTTPStatus.NOT_FOUND,
                )
            )
    else:
        prev_response = None

    if request.store:
        messages = _construct_input_messages(
            request=request, prev_response=prev_response
        )
        request.input = messages
        if previous_response_id:
            msg_store[previous_response_id] = messages
        else:
            msg_store[request.request_id] = messages
        # msg_store[previous_response_id] = messages
    tool_name_cache_dict = {}
    if isinstance(request.input, list) and isinstance(request.input[0], dict):
        new_input = []
        for idx, item in enumerate(request.input):
            new_item = copy.deepcopy(item)
            content = new_item.get("content")
            if content:
                if isinstance(content, list):
                    for i in new_item["content"]:
                        # if i["type"] == "text":
                        #     i["type"] = "input_text"
                        # if i["type"] == "image_url":
                        #     i["type"] = "input_image"
                        #     i["image_url"] = i["image_url"]["url"]
                        if i["type"] == "output_text" and new_item["type"] == "message":
                            new_item = {
                                "role": "assistant",
                                "content": i["text"],
                            }
                        if i["type"] == "input_image":
                            i["type"] = "image_url"
                            i["image_url"] = {"url": i["image_url"]}
                        if i["type"] == "input_text":
                            i["type"] = "text"

                        else:
                            pass
                new_input.append(new_item)
            else:
                type_ = item["type"]
                if type_ == "function_call":
                    tool_name_cache_dict[item["call_id"]] = item["name"]
                    arguments = item["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    new_item = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item["call_id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": arguments,
                                },
                            }
                        ],
                    }
                    new_input.append(new_item)
                elif type_ == "function_call_output":
                    tool_name = tool_name_cache_dict[item["call_id"]]
                    new_item = {
                        "role": "tool",
                        "name": tool_name,
                        "content": item["output"],
                    }
                    new_input.append(new_item)
                else:
                    new_input.append(item)
        request.input = new_input
    elif isinstance(request.input, str):
        # if request.instructions:
        #     messages = [{"role": "system", "content": request.instructions}]
        # messages = [{"role": "user", "content": request.input}]
        pass

    messages = request.input
    response_format = None
    if request.text:
        response_format = {}
        response_format["type"] = request.text.format.type
        if "json_schema" == request.text.format.type:
            response_format["json_schema"] = {"schema": request.text.format.schema_}

    reasoning_parser = None
    enable_thinking = True
    ## 转化工具
    tools = []
    for tool in request.tools:
        tool_dict = tool.model_dump()
        type_ = tool_dict.pop("type")
        tools.append({"type": type_, "function": tool_dict})
    # ----------- 需要进行转化映射 -----------
    gen_params = get_gen_params(
        request.model,
        "",
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        max_tokens=max_tokens,
        echo=False,
        stop=request.stop,
        tools=tools,
        tool_choice=request.tool_choice,
        response_format=response_format,
        reasoning_parser=reasoning_parser,
        enable_thinking=enable_thinking,
    )
    if gen_params["max_new_tokens"] is None:
        gen_params["max_new_tokens"] = 1024 * 16

    if request.stream:
        generator = responses_stream_generator(request, gen_params, worker_addr)
        content = _convert_stream_to_sse_events(generator=generator)
        return StreamingResponse(content, media_type="text/event-stream")

    content = await generate_completion(gen_params, worker_addr)
    if isinstance(content, str):
        content = json.loads(content)
    if content["error_code"] != 0:
        return create_error_response(content["error_code"], content["text"])
    output = []
    output_text = ResponseOutputText(
        text=content["text"],
        annotations=[],  # TODO
        type="output_text",
        logprobs=None,
    )
    message = ResponseOutputMessage(
        id=f"msg_{random_uuid()}",
        content=[output_text],
        role="assistant",
        status="completed",
        type="message",
    )

    tool_calls = content.get("tool_calls", None)
    if tool_calls:
        responses_tools = tool_calls2responses_tools(tool_calls)
        output.extend(responses_tools)
    else:
        output.append(message)
    created_time = int(time.time())
    response = ResponsesResponse.from_request(
        request,
        created_time=created_time,
        output=output,
        status="completed",
        usage=None,  # TODO
    )
    if request.store:
        async with response_store_lock:
            stored_response = response_store.get(response.id)
            # If the response is already cancelled, don't update it.
            if stored_response is None or stored_response.status != "cancelled":
                response_store[response.id] = response
    return response


def _construct_input_messages(
    request: ResponsesRequest,
    prev_response: ResponsesResponse | None = None,
) -> list:
    global response_store, msg_store
    messages = []
    if request.instructions:
        messages.append(
            {
                "role": "system",
                "content": request.instructions,
            }
        )
    # Prepend the conversation history.
    if prev_response is not None:
        prev_msg = msg_store.get(prev_response.id)
        if prev_msg:
            messages.extend(prev_msg)
        # Add the previous output.
        for output_item in prev_response.output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )
    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    else:
        messages.extend(request.input)  # type: ignore
    return messages


async def _convert_stream_to_sse_events(
    generator: AsyncGenerator["StreamingResponsesResponse", None],
) -> AsyncGenerator[str, None]:
    """Convert the generator to a stream of events in SSE format"""
    async for event in generator:
        event_type = getattr(event, "type", "unknown")
        # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
        event_data = (
            f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"
        )
        yield event_data


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
async def create_chat_completion(request: CustomChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = check_model(request.model)
    if error_check_ret is not None:
        return error_check_ret
    worker_addr = get_worker_address(request.model)
    max_tokens = 1024 * 8
    if request.max_completion_tokens:
        max_tokens = request.max_completion_tokens
    if request.max_tokens:
        max_tokens = request.max_tokens
    gen_params = get_gen_params(
        request.model,
        "",
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        max_tokens=max_tokens,
        echo=False,
        stop=request.stop,
        tools=request.tools,
        tool_choice=request.tool_choice,
        response_format=request.response_format,
        reasoning_parser=request.reasoning_parser,
        enable_thinking=request.enable_thinking,
    )
    if gen_params["max_new_tokens"] is None:
        gen_params["max_new_tokens"] = 1024 * 16

    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n, worker_addr
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    chat_completions = []
    for i in range(request.n):
        content = asyncio.create_task(generate_completion(gen_params, worker_addr))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if isinstance(content, str):
            content = json.loads(content)

        if content["error_code"] != 0:
            return create_error_response(content["error_code"], content["text"])
        choices.append(
            CustomChatCompletionResponseChoice(
                index=i,
                message=CustomChatMessage(
                    role="assistant",
                    content=content["text"],
                    tool_calls=content.get("tool_calls", None),
                ),
                finish_reason=content.get("finish_reason", "stop"),
            )
        )
        if "usage" in content:
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                if usage_value is None:
                    continue
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return CustomChatCompletionResponse(
        model=request.model, choices=choices, usage=usage
    )


from gpt_server.openai_api_protocol.custom_api_protocol import (
    CustomChatCompletionStreamResponse,
    CompletionResponse,
    CustomChatCompletionResponseStreamChoice,
    CustomDeltaMessage,
    StreamingResponsesResponse,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseContentPartAddedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
)
from contextlib import AsyncExitStack


async def _process_simple_streaming_events(
    request: ResponsesRequest,
    _increment_sequence_number_and_return: Callable[
        [StreamingResponsesResponse], StreamingResponsesResponse
    ],
    gen_params,
    worker_addr,
):
    global response_store, msg_store
    current_output_index = 0
    current_content_index = 0
    current_item_id = ""
    first_delta_sent = False
    if not first_delta_sent:
        current_item_id = str(uuid.uuid4())
    # TODO 这里暂时不考虑 reasoning_content
    yield _increment_sequence_number_and_return(
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=current_output_index,
            item=ResponseOutputMessage(
                id=current_item_id,
                type="message",
                role="assistant",
                content=[],
                status="in_progress",
            ),
        )
    )
    yield _increment_sequence_number_and_return(
        ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=-1,
            output_index=current_output_index,
            item_id=current_item_id,
            content_index=current_content_index,
            part=ResponseOutputText(
                type="output_text",
                text="",
                annotations=[],
                logprobs=[],
            ),
        )
    )
    current_content_index += 1
    first_delta_sent = True
    final_content = ""
    tool_calls = None
    async for content in generate_completion_stream(gen_params, worker_addr):
        try:
            error_code = content["error_code"]
        except Exception as e:
            logger.exception(f"发生异常 content：{content}")
            content["error_code"] = ErrorCode.INTERNAL_ERROR
        if content["error_code"] != 0:
            yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        delta_text = content.get("text", "")
        tool_calls = content.get("tool_calls", None)
        final_content += delta_text
        yield _increment_sequence_number_and_return(
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=-1,
                content_index=current_content_index,
                output_index=current_output_index,
                item_id=current_item_id,
                delta=delta_text,
                logprobs=[],
            )
        )
        current_content_index += 1
    yield _increment_sequence_number_and_return(
        ResponseTextDoneEvent(
            type="response.output_text.done",
            sequence_number=-1,
            output_index=current_output_index,
            content_index=current_content_index,
            text=final_content,
            logprobs=[],
            item_id=current_item_id,
        )
    )
    part = ResponseOutputText(
        text=final_content,
        type="output_text",
        annotations=[],
    )
    yield _increment_sequence_number_and_return(
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=-1,
            item_id=current_item_id,
            output_index=current_output_index,
            content_index=current_content_index,
            part=part,
        )
    )
    item = ResponseOutputMessage(
        type="message",
        role="assistant",
        content=[
            part,
        ],
        status="completed",
        id=current_item_id,
        summary=[],
    )
    yield _increment_sequence_number_and_return(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=current_output_index,
            item=item,
        )
    )

    output = []
    output_text = ResponseOutputText(
        text=final_content,
        annotations=[],  # TODO
        type="output_text",
        logprobs=None,
    )
    message = ResponseOutputMessage(
        id=f"msg_{random_uuid()}",
        content=[output_text],
        role="assistant",
        status="completed",
        type="message",
    )
    output.append(message)
    # tool_calls=[ChatCompletionMessageToolCall(id='chatcmpl-BeBhvTW4bjGu6JWJQz6s8k', function=Function(arguments='{"location": "南京", "unit": "celsius"}', name='get_weather'), type='function', index=None)]
    if tool_calls:
        responses_tools = tool_calls2responses_tools(tool_calls=tool_calls)
        output.extend(responses_tools)
    else:
        output.append(message)
    created_time = int(time.time())
    response = ResponsesResponse.from_request(
        request,
        created_time=created_time,
        output=output,
        status="completed",
        usage=None,  # TODO
    )
    if request.store:
        async with response_store_lock:
            stored_response = response_store.get(response.id)
            # If the response is already cancelled, don't update it.
            if stored_response is None or stored_response.status != "cancelled":
                response_store[response.id] = response
    yield _increment_sequence_number_and_return(
        ResponseCompletedEvent(
            type="response.completed",
            sequence_number=-1,
            response=response.model_dump(),
        )
    )


def tool_calls2responses_tools(tool_calls: list) -> list:
    responses_tools = []
    for tool in tool_calls:
        responses_tools.append(
            ResponseFunctionToolCall(
                type="function_call",
                arguments=tool["function"]["arguments"],
                call_id=tool["id"],
                name=tool["function"]["name"],
                id=tool["id"],
                status="completed",
            )
        )
    return responses_tools


async def responses_stream_generator(
    request: ResponsesRequest, gen_params: Dict[str, Any], worker_addr: str
) -> Generator[str, Any, None]:  # type: ignore
    # --------------------
    created_time = int(time.time())
    sequence_number = 0

    def _increment_sequence_number_and_return(
        event: StreamingResponsesResponse,
    ) -> StreamingResponsesResponse:
        nonlocal sequence_number
        # Set sequence_number if the event has this attribute
        if hasattr(event, "sequence_number"):
            event.sequence_number = sequence_number
        sequence_number += 1
        return event

    async with AsyncExitStack() as exit_stack:
        processer = None
        # TODO 没有考虑 use_harmony:
        processer = _process_simple_streaming_events
        initial_response = ResponsesResponse.from_request(
            request=request,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump()
        yield _increment_sequence_number_and_return(
            ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            )
        )
        yield _increment_sequence_number_and_return(
            ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            )
        )
        async for event_data in processer(
            request=request,
            _increment_sequence_number_and_return=_increment_sequence_number_and_return,
            gen_params=gen_params,
            worker_addr=worker_addr,
        ):
            yield event_data


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int, worker_addr: str
) -> Generator[str, Any, None]:  # type: ignore
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    for i in range(n):
        async for content in generate_completion_stream(gen_params, worker_addr):
            try:
                error_code = content["error_code"]
            except Exception as e:
                logger.exception(f"发生异常 content：{content}")
                content["error_code"] = ErrorCode.INTERNAL_ERROR
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            delta_text = content.get("text", "")
            choice_data = CustomChatCompletionResponseStreamChoice(
                index=i,
                delta=CustomDeltaMessage(
                    role="assistant",
                    content=delta_text,
                    tool_calls=content.get("tool_calls", None),
                    reasoning_content=content.get("reasoning_content", None),
                ),
                finish_reason=content.get("finish_reason", "stop"),
            )

            chunk = CustomChatCompletionStreamResponse(
                id=id,
                choices=[choice_data],
                model=model_name,
                usage=content.get("usage", None),
                created=int(time.time()),
                object="chat.completion.chunk",
            )
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.model_dump_json(exclude_unset=True)}\n\n"
    yield "data: [DONE]\n\n"


@app.post(
    "/v1/completions",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
async def create_completion(request: CompletionRequest):
    error_check_ret = check_model(request.model)
    if error_check_ret is not None:
        return error_check_ret

    request.prompt = process_input(request.model, request.prompt)

    worker_addr = get_worker_address(request.model)
    max_tokens = request.max_tokens
    for text in request.prompt:
        if isinstance(max_tokens, int) and max_tokens < request.max_tokens:
            request.max_tokens = max_tokens
    if request.stream:
        generator = generate_completion_stream_generator(
            request, request.n, worker_addr
        )
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        text_completions = []
        for text in request.prompt:
            gen_params = get_gen_params(
                request.model,
                worker_addr,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                max_tokens=request.max_tokens,
                logprobs=request.logprobs,
                echo=request.echo,
                stop=request.stop,
                best_of=request.best_of,
                use_beam_search=request.use_beam_search,
            )
            for i in range(request.n):
                content = asyncio.create_task(
                    generate_completion(gen_params, worker_addr)
                )
                text_completions.append(content)

        try:
            all_tasks = await asyncio.gather(*text_completions)
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

        choices = []
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])
            choices.append(
                CustomCompletionResponseChoice(
                    index=i,
                    text=content["text"],
                    logprobs=create_openai_logprobs(content.get("logprobs", None)),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            task_usage = UsageInfo.model_validate(content["usage"])
            for usage_key, usage_value in task_usage.model_dump().items():
                if usage_value is None:  # 不支持None的操作
                    continue
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.model_validate(usage)
        )


async def generate_completion_stream_generator(
    request: CompletionRequest, n: int, worker_addr: str
):
    model_name = request.model
    id = f"cmpl-{shortuuid.random()}"
    finish_stream_events = []
    for text in request.prompt:
        for i in range(n):
            previous_text = ""
            gen_params = get_gen_params(
                request.model,
                worker_addr,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                max_tokens=request.max_tokens,
                logprobs=request.logprobs,
                echo=request.echo,
                stop=request.stop,
            )
            async for content in generate_completion_stream(gen_params, worker_addr):
                if content["error_code"] != 0:
                    yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                decoded_unicode = content["text"].replace("\ufffd", "")
                delta_text = decoded_unicode[len(previous_text) :]
                previous_text = (
                    decoded_unicode
                    if len(decoded_unicode) > len(previous_text)
                    else previous_text
                )
                # todo: index is not apparent
                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text=delta_text,
                    logprobs=create_openai_logprobs(content.get("logprobs", None)),
                    finish_reason=content.get("finish_reason", None),
                )
                chunk = CompletionStreamResponse(
                    id=id,
                    object="text_completion",
                    choices=[choice_data],
                    model=model_name,
                )
                if len(delta_text) == 0:
                    if content.get("finish_reason", None) is not None:
                        finish_stream_events.append(chunk)
                    continue
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_completion_stream(payload: Dict[str, Any], worker_addr: str):
    async with httpx.AsyncClient() as client:
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=payload,
            timeout=60,
        ) as response:
            # content = await response.aread()
            buffer = b""
            async for raw_chunk in response.aiter_raw():
                buffer += raw_chunk
                while (chunk_end := buffer.find(delimiter)) >= 0:
                    chunk, buffer = buffer[:chunk_end], buffer[chunk_end + 1 :]
                    if not chunk:
                        continue
                    yield orjson.loads(chunk.decode())


async def generate_completion(payload: Dict[str, Any], worker_addr: str):
    return await fetch_remote(worker_addr + "/worker_generate", payload, "")


# TODO 使用CustomEmbeddingsRequest
from gpt_server.openai_api_protocol.custom_api_protocol import (
    CustomEmbeddingsRequest,
    RerankRequest,
    ModerationsRequest,
    SpeechRequest,
    OpenAISpeechRequest,
    ImagesGenRequest,
)


async def get_images_edits(payload: Dict[str, Any]):
    model_name = payload["model"]
    worker_addr = get_worker_address(model_name)

    transcription = await fetch_remote(
        worker_addr + "/worker_get_image_output", payload
    )
    return json.loads(transcription)


@app.post("/v1/images/edits", dependencies=[Depends(check_api_key)])
async def images_edits(
    model: str = Form(...),
    image: UploadFile = File(media_type="application/octet-stream"),
    prompt: Optional[Union[str, List[str]]] = Form(None),
    # negative_prompt: Optional[Union[str, List[str]]] = Form(None),
    response_format: Optional[str] = Form("url"),
    output_format: Optional[str] = Form("png"),
):
    """图片编辑"""

    error_check_ret = check_model(model)
    if error_check_ret is not None:
        return error_check_ret
    payload = {
        "image": base64.b64encode(await image.read()).decode(
            "utf-8"
        ),  # bytes → Base64 字符串,
        "model": model,
        "prompt": prompt,
        "output_format": output_format,
        "response_format": response_format,
    }
    result = await get_images_edits(payload=payload)
    return result


async def get_images_gen(payload: Dict[str, Any]):
    model_name = payload["model"]
    worker_addr = get_worker_address(model_name)

    transcription = await fetch_remote(
        worker_addr + "/worker_get_image_output", payload
    )
    return json.loads(transcription)


@app.post("/v1/images/generations", dependencies=[Depends(check_api_key)])
async def images_generations(request: ImagesGenRequest):
    """文生图"""
    error_check_ret = check_model(request.model)
    if error_check_ret is not None:
        return error_check_ret
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "output_format": request.output_format,
        "response_format": request.response_format,
    }
    result = await get_images_gen(payload=payload)
    return result


import edge_tts
import uuid

OUTPUT_DIR = "./edge_tts_cache"


async def generate_voice_stream(payload: Dict[str, Any], worker_addr: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            worker_addr,
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        ) as response:
            if response.status_code != 200:
                error_detail = await response.aread()
                raise Exception(f"API请求失败: {response.status_code},  {error_detail}")
            async for chunk in response.aiter_bytes():  # 流式迭代器
                yield chunk


@app.post("/v1/audio/speech", dependencies=[Depends(check_api_key)])
async def speech(request: OpenAISpeechRequest):
    controller_address = app_settings.controller_address
    error_check_ret = None
    models = await fetch_remote(controller_address + "/list_models", None, "models")
    if request.model not in models:
        error_check_ret = create_error_response(
            ErrorCode.INVALID_MODEL,
            f"Only {'&&'.join(models)} allowed now, your model {request.model}",
        )
    if error_check_ret is not None:
        return error_check_ret

    worker_addr = get_worker_address(request.model)
    response_format = request.response_format
    payload = {
        "model": request.model,
        "text": request.input,
        "response_format": response_format,
        "voice": request.voice,
        "speed": request.speed,
        "pitch": request.pitch,
    }
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(response_format, f"audio/{response_format}")
    if request.stream:
        stream_output = generate_voice_stream(
            payload, worker_addr + "/worker_generate_voice_stream"
        )
        return StreamingResponse(
            stream_output,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )


async def get_transcriptions(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    model_name = payload["model"]
    worker_addr = get_worker_address(model_name)

    transcription = await fetch_remote(
        worker_addr + "/worker_get_transcription", payload
    )
    return json.loads(transcription)


@app.post(
    "/v1/audio/transcriptions",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
async def transcriptions(file: UploadFile, model: str = Form()):
    controller_address = app_settings.controller_address
    error_check_ret = None
    models = await fetch_remote(controller_address + "/list_models", None, "models")
    if model not in models:
        error_check_ret = create_error_response(
            ErrorCode.INVALID_MODEL,
            f"Only {'&&'.join(models)} allowed now, your model {model}",
        )
    if error_check_ret is not None:
        return error_check_ret
    payload = {
        "model": model,
        "file": base64.b64encode(await file.read()).decode(
            "utf-8"
        ),  # bytes → Base64 字符串
        "language": "zh",
    }
    transcription = await get_transcriptions(payload)
    text = transcription["text"]
    return {"text": text}


@app.post(
    "/v1/moderations",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
async def classify(request: ModerationsRequest):
    error_check_ret = check_model(request.model)
    if error_check_ret is not None:
        return error_check_ret
    request.input = process_input(request.model, request.input)
    results = []
    token_num = 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE
    batches = [
        request.input[i : min(i + batch_size, len(request.input))]
        for i in range(0, len(request.input), batch_size)
    ]
    for num_batch, batch in enumerate(batches):
        payload = {
            "model": request.model,
            "input": batch,
            "threshold": request.threshold,
        }
        classify = await get_classify(payload)
        if "error_code" in classify and classify["error_code"] != 0:
            return create_error_response(classify["error_code"], classify["text"])
        for i, res in enumerate(classify["results"]):
            result = {
                "flagged": res["flagged"],
                "categories": res["categories"],
                "category_scores": res["category_scores"],
            }
            results.append(result)

        token_num += classify["token_num"]

    return {
        "id": shortuuid.random(),
        "model": request.model,
        "results": results,
    }


@app.post(
    "/v1/rerank",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
async def rerank(request: RerankRequest):
    error_check_ret = check_model(request.model)
    if error_check_ret is not None:
        return error_check_ret
    request.documents = process_input(request.model, request.documents)
    results = []
    token_num = 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE
    batches = [
        request.documents[i : min(i + batch_size, len(request.documents))]
        for i in range(0, len(request.documents), batch_size)
    ]
    for num_batch, batch in enumerate(batches):
        payload = {
            "model": request.model,
            "input": batch,
            "encoding_format": None,
            "query": request.query,  # TODO add query
        }
        embedding = await get_embedding(payload)
        if "error_code" in embedding and embedding["error_code"] != 0:
            return create_error_response(embedding["error_code"], embedding["text"])
        for i, emb in enumerate(embedding["embedding"]):
            result = {
                "index": num_batch * batch_size + i,
                "relevance_score": emb[0],
            }
            if request.return_documents:
                result["document"] = request.documents[num_batch * batch_size + i]
            results.append(result)

        token_num += embedding["token_num"]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    if request.top_n:
        results = results[: request.top_n]
    return {"results": results, "id": shortuuid.random()}


@app.post(
    "/v1/embeddings",
    dependencies=[Depends(check_api_key)],
    response_class=responses.ORJSONResponse,
)
async def create_embeddings(request: CustomEmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name
    error_check_ret = check_model(request.model)
    if error_check_ret is not None:
        return error_check_ret

    request.input = process_input(request.model, request.input)

    data = []
    token_num = 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE
    batches = [
        request.input[i : min(i + batch_size, len(request.input))]
        for i in range(0, len(request.input), batch_size)
    ]
    for num_batch, batch in enumerate(batches):
        payload = {
            "model": request.model,
            "input": batch,
            "encoding_format": request.encoding_format,
            "query": request.query,  # TODO add query
        }
        embedding = await get_embedding(payload)
        if "error_code" in embedding and embedding["error_code"] != 0:
            return create_error_response(embedding["error_code"], embedding["text"])
        data += [
            {
                "object": "embedding",
                "embedding": emb,
                "index": num_batch * batch_size + i,
            }
            for i, emb in enumerate(embedding["embedding"])
        ]
        token_num += embedding["token_num"]
    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).model_dump(exclude_none=True)


async def get_classify(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    model_name = payload["model"]
    worker_addr = get_worker_address(model_name)

    classify = await fetch_remote(worker_addr + "/worker_get_classify", payload)
    return json.loads(classify)


async def get_embedding(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    model_name = payload["model"]
    worker_addr = get_worker_address(model_name)

    embedding = await fetch_remote(worker_addr + "/worker_get_embeddings", payload)
    return json.loads(embedding)


### GENERAL API - NOT OPENAI COMPATIBLE ###


@app.post("/api/v1/token_check")
async def count_tokens(request: APITokenCheckRequest):
    """
    Checks the token count for each message in your list
    This is not part of the OpenAI API spec.
    """
    checkedList = []
    for item in request.prompts:
        worker_addr = get_worker_address(item.model)

        context_len = await fetch_remote(
            worker_addr + "/model_details",
            {"prompt": item.prompt, "model": item.model},
            "context_length",
        )

        token_num = await fetch_remote(
            worker_addr + "/count_token",
            {"prompt": item.prompt, "model": item.model},
            "count",
        )

        can_fit = True
        if token_num + item.max_tokens > context_len:
            can_fit = False

        checkedList.append(
            APITokenCheckResponseItem(
                fits=can_fit, contextLength=context_len, tokenCount=token_num
            )
        )

    return APITokenCheckResponse(prompts=checkedList)


def create_openai_api_server():
    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8082, help="port number")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        default=None,
        help="Optional list of comma separated API keys",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    os.environ["controller_address"] = args.controller_address
    if args.api_keys:
        os.environ["api_keys"] = args.api_keys

    logger.info(f"args: {args}")
    return args


if __name__ == "__main__":
    args = create_openai_api_server()
    if args.ssl:
        uvicorn.run(
            "gpt_server.serving.openai_api_server:app",
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
            workers=10,
        )
    else:
        uvicorn.run(
            "gpt_server.serving.openai_api_server:app",
            host=args.host,
            port=args.port,
            log_level="info",
            workers=10,
        )
