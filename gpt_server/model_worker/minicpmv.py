import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
import torch
from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase


class MiniCPMVWorker(ModelWorkerBase):
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
            model_type="AutoModel",
        )
        self.stop_words_ids = [
            151643,  # <|endoftext|>
            151645,  # <|im_end|>
        ]
        self.stop = [
            self.tokenizer.decode(skip_word) for skip_word in self.stop_words_ids
        ]
        logger.info(f"MiniCPMV停用词: {self.stop}")

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        logger.info(f"params {params}")
        logger.info(f"worker_id: {self.worker_id}")
        try:
            messages = params["messages"]
            if isinstance(messages, list):
                task = "chat"
            elif isinstance(messages, str):
                task = "completion"
            if task == "chat":
                text = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            elif task == "completion":
                text = messages

            input_ids = self.tokenizer([text], return_tensors="pt").input_ids
            params["messages"] = messages
            params["prompt"] = text
            params["stop"].extend(self.stop)
            params["stop_words_ids"] = self.stop_words_ids
            params["input_ids"] = input_ids

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

    def get_embeddings(self, params):
        return super().get_embeddings(params)


if __name__ == "__main__":
    MiniCPMVWorker.run()
