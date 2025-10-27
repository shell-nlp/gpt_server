import time
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union
import uuid

from pydantic import Field, BaseModel
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseOutputItem,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
    ResponseTextConfig,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    # ResponseReasoningPartAddedEvent,
    # ResponseReasoningPartDoneEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseStatus,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.tool import Tool
import shortuuid

ResponseInputOutputItem: TypeAlias = Union[
    ResponseInputItemParam, "ResponseReasoningItem", ResponseFunctionToolCall, Any
]

StreamingResponsesResponse: TypeAlias = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    # | ResponseReasoningPartAddedEvent
    # | ResponseReasoningPartDoneEvent
    | ResponseCodeInterpreterCallInProgressEvent
    | ResponseCodeInterpreterCallCodeDeltaEvent
    | ResponseWebSearchCallInProgressEvent
    | ResponseWebSearchCallSearchingEvent
    | ResponseWebSearchCallCompletedEvent
    | ResponseCodeInterpreterCallCodeDoneEvent
    | ResponseCodeInterpreterCallInterpretingEvent
    | ResponseCodeInterpreterCallCompletedEvent
)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    # only used to return cached tokens when --enable-cache-report is set
    prompt_tokens_details: Optional[Dict[str, int]] = None
    reasoning_tokens: Optional[int] = 0


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponseV2(BaseModel):
    error: ErrorInfo


class InputTokensDetails(BaseModel):
    cached_tokens: int
    input_tokens_per_turn: list[int] = Field(default_factory=list)
    cached_tokens_per_turn: list[int] = Field(default_factory=list)


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    tool_output_tokens: int = 0
    output_tokens_per_turn: list[int] = Field(default_factory=list)


class ResponseUsage(BaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class ResponseReasoningParam(BaseModel):
    """Reasoning parameters for responses."""

    effort: Optional[Literal["minimal", "low", "medium", "high"]] = Field(
        default="medium",
        description="Constrains effort on reasoning for reasoning models.",
    )


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: UsageInfo | None = None


class ResponsesRequest(BaseModel):
    """Request body for v1/responses endpoint."""

    # Core OpenAI API fields (ordered by official documentation)
    background: Optional[bool] = False
    include: Optional[
        List[
            Literal[
                "code_interpreter_call.outputs",
                "computer_call_output.output.image_url",
                "file_search_call.results",
                "message.input_image.image_url",
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
            ]
        ]
    ] = None
    input: Union[str, List[ResponseInputOutputItem]]
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    parallel_tool_calls: Optional[bool] = True
    previous_response_id: Optional[str] = None
    reasoning: Optional[ResponseReasoningParam] = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"
    store: Optional[bool] = True
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    text: ResponseTextConfig | None = None
    tool_choice: Literal["auto", "required", "none"] = "auto"
    tools: List[Tool] = Field(default_factory=list)
    top_logprobs: Optional[int] = 0
    top_p: Optional[float] = 1
    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None

    # Extra SGLang parameters
    request_id: str = Field(
        default_factory=lambda: f"resp_{uuid.uuid4().hex}",
        description="The request_id related to this request. If the caller does not set it, a random uuid will be generated.",
    )
    priority: int = Field(default=0, description="Request priority")
    extra_key: Optional[str] = Field(
        default=None,
        description="Extra key for classifying the request (e.g. cache_salt)",
    )
    cache_salt: Optional[str] = Field(
        default=None, description="Cache salt for request caching"
    )

    # SGLang-specific sampling parameters
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0


class ResponsesResponse(BaseModel):
    """Response body for v1/responses endpoint."""

    id: str = Field(default_factory=lambda: f"resp_{time.time()}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str

    output: List[
        Union[ResponseOutputItem, ResponseReasoningItem, ResponseFunctionToolCall]
    ] = Field(default_factory=list)
    status: Literal["queued", "in_progress", "completed", "failed", "cancelled"]
    usage: Optional[UsageInfo] = None
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    tools: List[Tool] = Field(default_factory=list)
    max_tool_calls: int | None = None
    # OpenAI compatibility fields. not all are used at the moment.
    # Recommend checking https://platform.openai.com/docs/api-reference/responses
    error: Optional[dict] = None
    incomplete_details: Optional[dict] = None  # TODO(v) support this input
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    reasoning: Optional[ResponseReasoningParam] = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"]
    store: Optional[bool] = None
    temperature: Optional[float] = None
    text: Optional[ResponseTextConfig] = None  # e.g. {"format": {"type": "text"}}
    top_logprobs: int | None = None
    top_p: Optional[float] = None
    truncation: Optional[str] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        created_time: int,
        output: list[ResponseOutputItem],
        status: ResponseStatus,
        usage: ResponseUsage | None = None,
    ) -> "ResponsesResponse":
        incomplete_details: IncompleteDetails | None = None
        if status == "incomplete":
            incomplete_details = IncompleteDetails(reason="max_output_tokens")
        # TODO: implement the other reason for incomplete_details,
        # which is content_filter
        # incomplete_details = IncompleteDetails(reason='content_filter')
        return cls(
            id=request.request_id,
            created_at=created_time,
            incomplete_details=incomplete_details,
            instructions=request.instructions,
            metadata=request.metadata,
            model=request.model,
            output=output,
            parallel_tool_calls=request.parallel_tool_calls,
            temperature=request.temperature,
            tool_choice=request.tool_choice,
            tools=request.tools,
            top_p=request.top_p,
            # background=request.background,
            max_output_tokens=request.max_output_tokens,
            max_tool_calls=request.max_tool_calls,
            previous_response_id=request.previous_response_id,
            reasoning=request.reasoning,
            service_tier=request.service_tier,
            status=status,
            text=request.text,
            top_logprobs=request.top_logprobs,
            truncation=request.truncation,
            user=request.user,
            usage=usage,
            store=request.store,
        )


class ImagesGenRequest(BaseModel):
    prompt: str
    model: str
    output_format: Literal["png", "jpeg", "webp"] = Field(
        default="png",
        description="png, jpeg, or webp",
    )
    # model_type: Literal["t2v", "t2i"] = Field(
    #     default="t2i",
    #     description="t2v: 文生视频 t2i: 文生图",
    # )
    response_format: Literal["url", "b64_json"] = Field(
        default="url",
        description="生成图像时返回的格式。必须为“ur”或“b64_json”之一。URL仅在图像生成后60分钟内有效。",
    )


# copy from https://github.com/remsky/Kokoro-FastAPI/blob/master/api/src/routers/openai_compatible.py
class OpenAISpeechRequest(BaseModel):
    model: str = Field(
        default=None,
        description="The model to use for generation.",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="新闻联播女声",
        description="暂时仅支持 新闻联播女声",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav, pcm. PCM format returns raw 16-bit samples without headers. AAC is not currently supported.",
    )
    stream: bool = Field(
        default=True,
        description="If true, audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = (
        Field(
            default="moderate",
            description="Specifies the pitch level for the generated audio. Valid options: 'very_low', 'low', 'moderate', 'high', 'very_high'.",
        )
    )
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = (
        Field(
            default="moderate",
            description="Specifies the speed level of the audio output. Valid options: 'very_low', 'low', 'moderate', 'high', 'very_high'.",
        )
    )


class SpeechRequest(BaseModel):
    "TTS"

    model: str = Field(
        default="edge_tts", description="One of the available TTS models:"
    )
    input: str = Field(
        description="The text to generate audio for. The maximum length is 4096 characters."
    )
    voice: str = Field(
        default="zh-CN-YunxiNeural",
        description="The voice to use when generating the audio",
    )
    response_format: Optional[str] = Field(
        default="mp3", description="The format of the audio"
    )
    speed: Optional[float] = Field(
        default=1.0,
        description="The speed of the generated audio. Select a value from 0.25 to 5.0. 1.0 is the default.",
        ge=0,
        le=5,
    )


class ModerationsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    threshold: float = Field(default=0.5, description="审核的阈值")


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False
    # max_chunks_per_doc: Optional[int] = Field(default=None, alias="max_tokens_per_doc")


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class CustomModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []
    owned_by: str = "gpt_server"


class ModelList(BaseModel):
    object: str = "list"
    data: List[CustomModelCard] = []


class CustomEmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None
    encoding_format: Optional[str] = None
    query: Optional[str] = None


class CustomChatCompletionRequest(BaseModel):
    model: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[Literal["none"], Literal["auto"], Any]] = "auto"
    messages: Union[
        str,
        List[dict],
        # List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    response_format: Optional[Any] = None
    reasoning_parser: Optional[str] = None
    max_completion_tokens: Optional[int] = None
    enable_thinking: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str


class CustomChatMessage(ChatMessage):
    tool_calls: Optional[list] = None


class CustomChatCompletionResponseChoice(BaseModel):
    index: int
    message: CustomChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "error"]] = None


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CustomCompletionResponseChoice(BaseModel):
    """completion 的响应结构"""

    index: int
    text: str
    logprobs: Optional[LogProbs] = None

    finish_reason: Optional[Literal["stop", "length", "tool_calls", "error"]] = None


class CustomChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    usage: UsageInfo
    choices: List[CustomChatCompletionResponseChoice]


# chat.completion.chunk
class CustomDeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    reasoning_content: Optional[str] = None


class CustomChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: CustomDeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "error"]] = None


class CustomChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    usage: Optional[UsageInfo] = Field(default=None)
    choices: List[CustomChatCompletionResponseStreamChoice]


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CustomCompletionResponseChoice]
    usage: UsageInfo
