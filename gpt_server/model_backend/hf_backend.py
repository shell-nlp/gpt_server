from typing import Any, Dict
import torch
import os
import json
from peft import PeftModel
from transformers import TextIteratorStreamer
from transformers.generation.logits_process import LogitsProcessorList
from threading import Thread
from gpt_server.model_backend.base import ModelBackend
from gpt_server.model_backend.utils import (
    InvalidScoreLogitsProcessor,
    StoppingCriteriaList,
    StopAtSpecificTokenCriteria,
    XgrammarLogitsProcessor,
)
import asyncio
from loguru import logger

invalid_score_processor = InvalidScoreLogitsProcessor()


class NoneContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class HFBackend(ModelBackend):
    def __init__(self, tokenizer, model: torch.nn.Module) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.xgrammar_processor = XgrammarLogitsProcessor(tokenizer)
        self.lora_requests = []
        lora = os.getenv("lora", None)
        if lora:
            lora_dict: dict = json.loads(lora)
            for i, (lora_name, lora_path) in enumerate(lora_dict.items()):
                self.lora_requests.append(
                    dict(
                        lora_name=lora_name,
                        lora_int_id=i,
                        lora_local_path=lora_path,
                    )
                )
                if i == 0:
                    self.model = PeftModel.from_pretrained(
                        model=model,
                        model_id=lora_path,
                        adapter_name=lora_name,
                    )
                    continue
                self.model.load_adapter(model_id=lora_path, adapter_name=lora_name)

    async def stream_chat(self, params: Dict[str, Any]):
        prompt = params.get("prompt", "")
        logger.info(prompt)
        temperature = float(params.get("temperature", 0.8))
        top_p = float(params.get("top_p", 0.8))
        max_new_tokens = int(params.get("max_new_tokens", 512))
        # top_k = params.get("top_k", -1.0)
        # TODO ValueError: The following `model_kwargs` are not used by the model: ['presence_penalty', 'frequency_penalty'] (note: typos in the generate arguments will also show up in this list)
        # presence_penalty = float(params.get("presence_penalty", 0.0))
        # frequency_penalty = float(params.get("frequency_penalty", 0.0))
        stop = params.get("stop", [])  # 停止的 token
        input_ids = params.get("input_ids", None)
        if input_ids is None:
            input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
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
        # TODO
        # ---- 支持 response_format,但是官方对BPE分词器的支持仍然太差 ----
        response_format = params["response_format"]
        if response_format is not None:
            if response_format["type"] == "json_object":
                xgrammar_processor = (
                    self.xgrammar_processor.get_json_grammar_processor()
                )
                logits_processor.append(xgrammar_processor)

            elif response_format["type"] == "json_schema":
                json_schema = response_format["json_schema"]
                assert json_schema is not None
                guided_json = json_schema["schema"]
                xgrammar_processor = self.xgrammar_processor.get_json_schema_processor(
                    schema=json.dumps(guided_json)
                )
                logits_processor.append(xgrammar_processor)
            elif response_format["type"] == "text":
                pass

        # ---- 支持 response_format,但是官方对BPE分词器的支持仍然太差 ----
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
        use_lora = False
        for lora in self.lora_requests:
            if params["model"] == lora["lora_name"]:
                self.model.set_adapter(lora["lora_name"])
                use_lora = True
                break
        context_manager = NoneContextManager()
        if not use_lora and self.lora_requests:
            context_manager = self.model.disable_adapter()
        with context_manager:
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
        prompt_tokens = len(input_ids.tolist()[0])
        completion_tokens = 0
        stop_flag = False
        try:
            current_text = ""
            previous_text = ""
            previous_token_ids = []
            current_token_ids = []
            delta_token_ids = []
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
                current_text = current_text + new_text
                completion_tokens += 1
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                ret = {
                    "text": new_text,
                    "error_code": 0,
                    "usage": usage,
                }
                yield ret
                if stop_flag:
                    break
                # 用来解决输出卡顿的问题
                await asyncio.sleep(0.02)
            logger.info(current_text)
        except asyncio.CancelledError as e:
            stop_specific_token_criteria.stop = True
