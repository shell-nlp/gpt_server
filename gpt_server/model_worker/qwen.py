import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
import torch
from gpt_server.model_handler.qwen import (
    conv2messages,
    make_context,
    get_stop_words_ids,
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

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        print("params", params)
        print("worker_id:", self.worker_id)
        try:
            prompt = params["prompt"]
            query, messages = conv2messages(prompt=prompt)
            raw_text, context_tokens = make_context(
                tokenizer=self.tokenizer, query=query, history=None, system=""
            )
            stop_words_ids = get_stop_words_ids(
                tokenizer=self.tokenizer, chat_format="chatml"
            )
            skip_word_list = [self.tokenizer.im_end_id, self.tokenizer.im_start_id, self.tokenizer.eos_token_id or 151643]
            params["stop_words_ids"] = stop_words_ids
            input_ids = torch.tensor([context_tokens])
            params["input_ids"] = input_ids
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
