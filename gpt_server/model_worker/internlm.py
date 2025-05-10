import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
import torch
from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase


class InternlmWorker(ModelWorkerBase):
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
            1,  # bos
            2,  # eos
            92543,  # <|im_start|>
            92542,  # <|im_end|>
        ]

        self.stop = [
            self.tokenizer.decode(skip_word) for skip_word in self.stop_words_ids
        ]
        logger.warning(f"{model_names[0]} 停用词: {self.stop}")
        self.other_config = {
            "chat_template": "{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        }

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        try:
            model_type = getattr(self.model_config, "model_type", "internlm")
            messages = params["messages"]
            if isinstance(messages, list):
                task = "chat"
                for msg in messages:
                    if msg["role"] == "function":
                        msg["role"] = "Observation:"
            elif isinstance(messages, str):
                task = "completion"
            if task == "chat":
                # 暂时保留，用于特殊情况的处理
                if model_type == "internlm":
                    logger.info("正在使用internlm-1.0 !")
                    text = self.tokenizer.apply_chat_template(
                        conversation=messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template=self.other_config["chat_template"],
                    )
                elif model_type == "internlm2":
                    logger.info("正在使用internlm-2.0 !")
                    text = self.tokenizer.apply_chat_template(
                        conversation=messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            elif task == "completion":
                text = messages

            # ---------------添加额外的参数------------------------
            params["messages"] = messages
            params["prompt"] = text
            params["stop"].extend(self.stop)
            params["stop_words_ids"] = self.stop_words_ids
            # ---------------添加额外的参数------------------------
            async for ret in self.backend.stream_chat(params=params):
                response = ret["text"]

                yield json.dumps(ret).encode() + b"\0"

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            logger.info(e)
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"


if __name__ == "__main__":
    InternlmWorker.run()
