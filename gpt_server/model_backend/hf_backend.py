from typing import Any, Dict
import torch.nn as nn
from transformers import TextIteratorStreamer
from threading import Thread
from gpt_server.model_backend.base import ModelBackend


class HFBackend(ModelBackend):
    def __init__(self, tokenizer, model: nn.Module) -> None:
        self.model = model
        self.tokenizer = tokenizer

    async def stream_chat(self, query: str, params: Dict[str, Any]):
        # context = params.pop("prompt")
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        max_new_tokens = int(params.get("max_new_tokens", 512))
        # top_k = params.get("top_k", -1.0)
        # TODO ValueError: The following `model_kwargs` are not used by the model: ['presence_penalty', 'frequency_penalty'] (note: typos in the generate arguments will also show up in this list)
        # presence_penalty = float(params.get("presence_penalty", 0.0))
        # frequency_penalty = float(params.get("frequency_penalty", 0.0))
        input_ids = params.get("input_ids")
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            decode_kwargsl={"skip_special_tokens": True},
        )
        generation_kwargs = dict(
            input_ids=input_ids.to(self.model.device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            # top_k=top_k,
            # presence_penalty=presence_penalty,
            # frequency_penalty=frequency_penalty,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        prompt_tokens = len(input_ids.tolist()[0])
        completion_tokens = 0
        for new_text in streamer:
            completion_tokens += 1
            generated_text += new_text
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            yield generated_text, usage
