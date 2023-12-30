import os
import json
from typing import List
from fastchat.serve.base_model_worker import BaseModelWorker, app
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from loguru import logger
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig

import uuid
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from fastchat.utils import (
    get_context_length,
)

import torch
from gpt_server.model_handler.chatglm3 import conv2messages
from gpt_server.utils import get_free_tcp_port


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


invalid_score_processor = InvalidScoreLogitsProcessor()


class ChatGLM3Worker(BaseModelWorker):
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

        # TODO ----------------------------- hf 模型加载------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            encode_special_tokens=True,
        )
        use_deepspeed = os.getenv("USE_DS", 0)
        use_accelerate = os.getenv("USE_ACC",0)
      
        if use_accelerate and use_deepspeed:
            assert 0, "ds 和 acc 不能同时设置为 True"
        if not use_deepspeed and not use_accelerate:
            logger.info("使用hf")
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=None if use_deepspeed else "auto",
            ).half()
            self.model = self.model.eval()
        # TODO -----------------------------hf 模型加载------------------------
        # TODO -----------------------------DeepSpeed 模型加载------------------------
        if use_deepspeed:
            from gpt_server.model_backend.ds_worker import get_ds_model

            logger.info("使用deepspeed")
            ds_model = get_ds_model(model_path=model_path)
            self.model = ds_model
        if use_accelerate:
            from gpt_server.model_backend.acc_worker import get_acc_model

            logger.info("使用accelerate")
            acc_model = get_acc_model(model_path=model_path)
            self.model = acc_model
        # TODO -----------------------------DeepSpeed 模型加载------------------------

        self.context_len = get_context_length(self.model.config)
        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.init_heart_beat()

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

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())


def get_worker(
    model_path: str,
    controller_addr: str = "http://localhost:21001",
    worker_addr: str = "http://localhost:21002",
    worker_id: str = str(uuid.uuid4())[:8],
    model_names: List[str] = ["chatglm3-6b-2"],
    limit_worker_concurrency: int = 6,
    conv_template: str = None,  # type: ignore
):
    worker = ChatGLM3Worker(
        controller_addr,
        worker_addr,
        worker_id,
        model_path,
        model_names,
        limit_worker_concurrency,
        conv_template=conv_template,
    )
    return worker


def main():
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="gpus")

    parser.add_argument("--local_rank", type=str, default="local-rank")  # 必传
    parser.add_argument("--master_port", type=str, default="master_port")
    parser.add_argument("--model_name_or_path", type=str, default="model_name_or_path")
    parser.add_argument(
        "--model_names", type=lambda s: s.split(","), default="model_names"
    )

    args = parser.parse_args()
    use_deepspeed = os.getenv("USE_DS", 0)
    if use_deepspeed:
        print("local-rank", args.local_rank)
        print("master_port", args.master_port)
        os.environ["MASTER_PORT"] = args.master_port
        # DS 只能在内部生效
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    host = "localhost"
    port = get_free_tcp_port()
    worker_addr = f"http://{host}:{port}"
    worker = get_worker(
        worker_addr=worker_addr,
        model_path=args.model_name_or_path,
        model_names=args.model_names,
    )
    if args.local_rank == "0":
        print("=======================================")
        print(f"{args.model_names[0]} 启动成功!")
        print("=======================================")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
