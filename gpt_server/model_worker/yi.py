import json
from threading import Thread
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from transformers import GenerationConfig, TextIteratorStreamer
import torch
from gpt_server.model_handler.yi import conv2messages

from gpt_server.model_worker.base import ModelWorkerBase


class YiWorker(ModelWorkerBase):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str = None,  # type: ignore
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
            model_type="LlamaForCausalLM",
        )

    def load_model_tokenizer(self, model_path):
        return super().load_model_tokenizer(model_path)

    def generate_stream_gate(self, params):
        self.call_ct += 1
        print("params", params)
        print("worker_id:", self.worker_id)
        try:
            prompt = params["prompt"]
            temperature = float(params.get("temperature", 0.6))
            top_p = float(params.get("top_p", 0.8))
            max_new_tokens = int(params.get("max_new_tokens", 512))
            messages = conv2messages(prompt=prompt)
            print(1, messages)
            input_ids = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            print(self.model.device)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                decode_kwargsl={"skip_special_tokens": True},
            )
            generation_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=7,
                bos_token_id=6,
                pad_token_id=0,
            )
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            generated_text = ""
            special_tokens = self.tokenizer.special_tokens_map[
                "additional_special_tokens"
            ]
            for new_text in streamer:
                # 针对 yi模型 特别处理
                if "<|im_end|>" in new_text:
                    idx = new_text.rfind("<|im_end|>")
                    new_text = new_text[:idx]

                generated_text += new_text

                ret = {
                    "text": generated_text,
                    "error_code": 0,
                }

                yield json.dumps(ret).encode() + b"\0"

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            print(e)
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def get_embeddings(self, params):
        return super().get_embeddings(params)


if __name__ == "__main__":
    YiWorker.run()
