from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelBackend(ABC):
    @abstractmethod
    def stream_chat(self, params: Dict[str, Any]):
        pass
