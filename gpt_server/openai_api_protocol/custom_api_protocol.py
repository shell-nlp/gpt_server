from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any, Union
from fastchat.protocol.openai_api_protocol import EmbeddingsRequest

class CustomEmbeddingsRequest(EmbeddingsRequest):
    query:Optional[str] = None
