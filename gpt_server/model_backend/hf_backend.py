from typing import Any, Dict
import torch.nn as nn
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
from gpt_server.model_backend.base import ModelBackend


class HFBackend(ModelBackend):
    def __init__(self, model_path: str, model: nn.Module) -> None:
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def stream_chat(self, query: str, params: Dict[str, Any]):
        top_p = params.get("top_p")
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")
        input_ids = params.get("input_ids")
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            decode_kwargsl={"skip_special_tokens": True},
        )
        generation_kwargs = dict(
            input_ids=input_ids.to(self.model.device),
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text
