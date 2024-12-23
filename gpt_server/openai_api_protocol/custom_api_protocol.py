from typing import Any, Dict, List, Literal, Optional, Union
from fastchat.protocol.openai_api_protocol import (
    EmbeddingsRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionResponseStreamChoice,
    UsageInfo,
    DeltaMessage,
    ModelCard,
)
from pydantic import Field, BaseModel


class SpeechRequest(BaseModel):
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
        description="The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.",
    )


class ModerationsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False
    # max_chunks_per_doc: Optional[int] = Field(default=None, alias="max_tokens_per_doc")


class CustomModelCard(ModelCard):
    owned_by: str = "gpt_server"


class CustomEmbeddingsRequest(EmbeddingsRequest):
    query: Optional[str] = None


class CustomChatCompletionRequest(ChatCompletionRequest):
    tools: Optional[list] = None
    tool_choice: Optional[Union[Literal["none"], Literal["auto"], Any]] = "none"
    messages: Union[
        str,
        List[dict],
        # List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    response_format: Optional[Any] = None


class CustomChatMessage(ChatMessage):
    tool_calls: Optional[list] = None


class CustomChatCompletionResponseChoice(ChatCompletionResponseChoice):
    message: CustomChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class CustomChatCompletionResponse(ChatCompletionResponse):
    choices: List[CustomChatCompletionResponseChoice]


# chat.completion.chunk
class CustomDeltaMessage(DeltaMessage):
    tool_calls: Optional[list] = None


class CustomChatCompletionResponseStreamChoice(ChatCompletionResponseStreamChoice):
    delta: CustomDeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class CustomChatCompletionStreamResponse(ChatCompletionStreamResponse):
    usage: Optional[UsageInfo] = Field(default=None)
    choices: List[CustomChatCompletionResponseStreamChoice]
