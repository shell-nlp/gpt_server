from typing import Optional
from fastchat.protocol.openai_api_protocol import EmbeddingsRequest


class CustomEmbeddingsRequest(EmbeddingsRequest):
    query: Optional[str] = None
