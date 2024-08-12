from typing import Any, Dict
import torch
from transformers import TextIteratorStreamer
from transformers.generation.logits_process import LogitsProcessorList
from threading import Thread
from gpt_server.model_backend.base import ModelBackend
from gpt_server.model_backend.utils import (
    InvalidScoreLogitsProcessor,
    StoppingCriteriaList,
    StopAtSpecificTokenCriteria,
)
import asyncio
from loguru import logger

invalid_score_processor = InvalidScoreLogitsProcessor()


class HFBackend(ModelBackend):
    def __init__(self, tokenizer, model: torch.nn.Module) -> None:
        self.model = model
        self.tokenizer = tokenizer

    async def stream_chat(self, params: Dict[str, Any]):
        prompt = params.get("prompt","")
        logger.info(prompt)
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        max_new_tokens = int(params.get("max_new_tokens", 512))
        # top_k = params.get("top_k", -1.0)
        # TODO ValueError: The following `model_kwargs` are not used by the model: ['presence_penalty', 'frequency_penalty'] (note: typos in the generate arguments will also show up in this list)
        # presence_penalty = float(params.get("presence_penalty", 0.0))
        # frequency_penalty = float(params.get("frequency_penalty", 0.0))
        stop = params.get("stop", [])  # 停止的 token
        input_ids = params.get("input_ids")
        stop_words_ids = params.get("stop_words_ids", [])
        if temperature <= 1e-5:
            top_p = 1.0
            temperature = 0.01

        stopping_criteria = StoppingCriteriaList()  # 停止条件
        stop_specific_token_criteria = StopAtSpecificTokenCriteria(
            token_id_list=stop_words_ids
        )
        stopping_criteria.append(stop_specific_token_criteria)
        logits_processor = LogitsProcessorList([invalid_score_processor])
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
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            # top_k=top_k,
            # presence_penalty=presence_penalty,
            # frequency_penalty=frequency_penalty,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        prompt_tokens = len(input_ids.tolist()[0])
        completion_tokens = 0
        stop_flag = False
        try:
            for new_text in streamer:
                for stop_word in stop:
                    if stop_word in new_text:
                        idx = new_text.rfind(stop_word)
                        stop_flag = True
                        print(
                            "********** 停止的单词为:",
                            stop_word,
                            "in",
                            new_text,
                            "**********",
                        )
                        new_text = new_text[:idx]
                        break
                completion_tokens += 1
                generated_text += new_text
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                ret = {
                    "text": generated_text,
                    "error_code": 0,
                    "usage": usage,
                }
                yield ret
                if stop_flag:
                    break
                # 用来解决输出卡顿的问题
                await asyncio.sleep(0.02)
            logger.info(generated_text)
        except asyncio.CancelledError as e:
            stop_specific_token_criteria.stop = True
