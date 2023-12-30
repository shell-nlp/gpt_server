import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from transformers.generation.logits_process import LogitsProcessor
import torch
from gpt_server.model_handler.chatglm3 import conv2messages
from gpt_server.model_worker.base import ModelWorkerBase


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


invalid_score_processor = InvalidScoreLogitsProcessor()


class ChatGLM3Worker(ModelWorkerBase):
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
        )

    def load_model_tokenizer(self, model_path):
        return super().load_model_tokenizer(model_path)

    def generate_stream_gate(self, params):
        self.call_ct += 1
        print("params", params)
        print("worker_id:", self.worker_id)
        try:
            prompt = params["prompt"]
            temperature = float(params.get("temperature", 0.8))
            top_p = float(params.get("top_p", 0.8))
            max_new_tokens = int(params.get("max_new_tokens", 512))

            query, messages = conv2messages(prompt=prompt)

            for response, new_history in self.model.stream_chat(
                tokenizer=self.tokenizer,
                query=query,
                history=messages if messages else None,
                past_key_values=None,
                max_length=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                logits_processor=[invalid_score_processor],
                return_past_key_values=False,
            ):
                ret = {
                    "text": response,
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
    ChatGLM3Worker.run()
