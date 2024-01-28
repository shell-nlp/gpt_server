from typing import Any, Dict
import torch.nn as nn
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
from gpt_server.model_backend.base import ModelBackend


class HFBackend(ModelBackend):
    def __init__(self, tokenizer, model: nn.Module) -> None:
        self.model = model
        self.tokenizer = tokenizer

    async def stream_chat(self, query: str, params: Dict[str, Any]):
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        max_tokens = int(params.get("max_new_tokens", 512))
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
        thread.join()
