from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelBackend(ABC):
    @abstractmethod
    def stream_chat(self, query: str, params: Dict[str, Any]):
        pass
