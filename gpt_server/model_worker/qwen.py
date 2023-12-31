import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from transformers.generation.logits_process import LogitsProcessor
from transformers import GenerationConfig
import torch
from gpt_server.model_handler.qwen import conv2messages

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
            print(1, query)
            print(2, messages)
            stream_generator = self.model.chat_stream(
                tokenizer=self.tokenizer,
                query=query,
                history=messages if messages else None,
                system="You are a helpful assistant.",
                # stop_words_ids=None,
                # logits_processor=None,
                generation_config=GenerationConfig(
                    # temperature=temperature,
                    chat_format="chatml",
                    eos_token_id = 151643,
                    pad_token_id=151643,
                    max_window_size=6144,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k = 0,
                    top_p=top_p,
                    repetition_penalty = 1.1,
                    transformers_version="4.31.0"
                ),
            )
            for text in stream_generator:
                ret = {
                    "text": text,
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
    QwenWorker.run()
