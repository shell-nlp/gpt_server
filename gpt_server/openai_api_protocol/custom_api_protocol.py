from typing import List, Literal, Optional, Union
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
)
from pydantic import Field


class CustomEmbeddingsRequest(EmbeddingsRequest):
    query: Optional[str] = None


class CustomChatCompletionRequest(ChatCompletionRequest):
    tools: Optional[list] = None
    tool_choice: Optional[dict] = None


class CustomChatMessage(ChatMessage):
    tool_calls: Optional[list] = None


class CustomChatCompletionResponseChoice(ChatCompletionResponseChoice):
    message: CustomChatMessage


class CustomChatCompletionResponse(ChatCompletionResponse):
    choices: List[CustomChatCompletionResponseChoice]


# chat.completion.chunk
class CustomDeltaMessage(DeltaMessage):
    tool_calls: Optional[list] = None


class CustomChatCompletionResponseStreamChoice(ChatCompletionResponseStreamChoice):
    delta: CustomDeltaMessage


class CustomChatCompletionStreamResponse(ChatCompletionStreamResponse):
    usage: Optional[UsageInfo] = Field(default=None)
    choices: List[CustomChatCompletionResponseStreamChoice]
