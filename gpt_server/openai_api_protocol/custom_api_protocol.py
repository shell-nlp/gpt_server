from typing import Optional
from fastchat.protocol.openai_api_protocol import (
    EmbeddingsRequest,
    ChatCompletionRequest,
)


class CustomEmbeddingsRequest(EmbeddingsRequest):
    query: Optional[str] = None


class CustomChatCompletionRequest(ChatCompletionRequest):
    tools: Optional[list] = None
