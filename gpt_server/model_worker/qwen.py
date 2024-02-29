import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
import torch
from gpt_server.model_handler.qwen import (
    conv2messages,
    make_context,
)

from gpt_server.model_worker.base import ModelWorkerBase


class QwenWorker(ModelWorkerBase):
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
            model_type="AutoModelForCausalLM",
        )

        self.stop_words_ids = [
            151643,  # <|endoftext|>
            151644,  # <|im_start|>
            151645,  # <|im_end|>
        ]

        self.stop = [
            self.tokenizer.decode(skip_word) for skip_word in self.stop_words_ids
        ]
        print("qwen停用词:", self.stop)
        self.other_config = {
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        }

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        print("params", params)
        print("worker_id:", self.worker_id)
        try:
            model_type = getattr(self.model_config, "model_type", "qwen")
            query = ""
            messages = params["messages"]
            for msg in messages:
                if msg["role"] == "function":
                    msg["role"] = "Observation:"
            # 暂时保留，用于特殊情况的处理
            if model_type == "qwen":
                print("正在使用qwen-1.0 !")
                text = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=self.other_config["chat_template"],
                )
                input_ids = self.tokenizer([text], return_tensors="pt").input_ids
            elif model_type == "qwen2":
                print("正在使用qwen-2.0 !")
                text = self.tokenizer.apply_chat_template(
                    conversation=messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = self.tokenizer([text], return_tensors="pt").input_ids
            print(self.tokenizer.decode(input_ids.tolist()[0]))
            # ---------------添加额外的参数------------------------
            params["stop"].extend(self.stop)
            params["stop_words_ids"] = self.stop_words_ids
            params["input_ids"] = input_ids
            # ---------------添加额外的参数------------------------
            async for response, usage in self.backend.stream_chat(
                query=query, params=params
            ):
                ret = {"text": response, "error_code": 0, "usage": usage}

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
    QwenWorker.run()
