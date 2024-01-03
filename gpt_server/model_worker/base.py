from typing import List
import json
from abc import ABC, abstractmethod
from fastchat.serve.base_model_worker import BaseModelWorker, app
from fastchat.utils import (
    get_context_length,
)
from loguru import logger
import os
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
)
import torch
import uuid
from gpt_server.utils import get_free_tcp_port


class ModelWorkerBase(BaseModelWorker, ABC):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str = None,  # type: ignore
        model_type: str = "AutoModel",
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
        os.environ["WORKER_NAME"] = self.__class__.__name__
        self.use_deepspeed = os.getenv("USE_DS", 0)
        self.use_accelerate = os.getenv("USE_ACC", 0)
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.load_model_tokenizer(model_path)
        self.context_len = self.get_context_length()
        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.init_heart_beat()

    def get_context_length(
        self,
    ):
        """ "支持的最大 token 长度"""
        if self.model is None:
            return 512
        return get_context_length(self.model.config)

    def get_model_class(self):
        MODEL_CLASS = AutoModel
        if self.model_type == "LlamaForCausalLM":
            MODEL_CLASS = LlamaForCausalLM
            register = AutoModelForCausalLM._model_mapping.register
            register(LlamaForCausalLM.config_class, LlamaForCausalLM, exist_ok=True)
            MODEL_CLASS = AutoModelForCausalLM

        elif self.model_type == "AutoModel":
            MODEL_CLASS = AutoModel
        elif self.model_type == "AutoModelForCausalLM":
            MODEL_CLASS = AutoModelForCausalLM

        return MODEL_CLASS

    @abstractmethod
    def load_model_tokenizer(self, model_path):
        """加载 模型 和 分词器 直接对 self.model 和 self.tokenizer 进行赋值"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            encode_special_tokens=True,
        )

        if self.use_accelerate and self.use_deepspeed:
            assert 0, "ds 和 acc 不能同时设置为 True"

        MODEL_CLASS = self.get_model_class()
        if not self.use_deepspeed and not self.use_accelerate:
            logger.info("使用hf")
            self.model = MODEL_CLASS.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=None if self.use_deepspeed else "auto",
            ).half()

            self.model = self.model.eval()

        if self.use_deepspeed:
            from gpt_server.model_backend.deepspeed_backend import get_ds_model

            logger.info("使用deepspeed")
            ds_model = get_ds_model(model_path=model_path, model_class=MODEL_CLASS)
            self.model = ds_model
        if self.use_accelerate:
            from gpt_server.model_backend.accelerate_backend import get_acc_model

            logger.info("使用accelerate")
            acc_model = get_acc_model(model_path=model_path, model_class=MODEL_CLASS)
            self.model = acc_model

    @abstractmethod
    def generate_stream_gate(self, params):
        pass

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    @abstractmethod
    def get_embeddings(self, params):
        pass

    @classmethod
    def get_worker(
        cls,
        model_path: str,
        controller_addr: str = "http://localhost:21001",
        worker_addr: str = "http://localhost:21002",
        worker_id: str = str(uuid.uuid4())[:8],
        model_names: List[str] = ["chatglm3-6b-2"],
        limit_worker_concurrency: int = 6,
        conv_template: str = None,  # type: ignore
    ):
        worker = cls(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )
        return worker

    @classmethod
    def run(cls):
        import uvicorn
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--gpus", type=str, default="gpus")

        parser.add_argument("--local_rank", type=str, default="local-rank")  # 必传
        parser.add_argument("--master_port", type=str, default="master_port")
        parser.add_argument(
            "--model_name_or_path", type=str, default="model_name_or_path"
        )
        parser.add_argument(
            "--model_names", type=lambda s: s.split(","), default="model_names"
        )

        args = parser.parse_args()
        use_deepspeed = os.getenv("USE_DS", 0)
        if use_deepspeed:
            print("local-rank", args.local_rank)
            print("master_port", args.master_port)
            os.environ["Local_RANK"] = args.local_rank
            os.environ["MASTER_PORT"] = args.master_port
            # DS 只能在内部生效
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

        host = "localhost"
        port = get_free_tcp_port()
        worker_addr = f"http://{host}:{port}"
        worker = cls.get_worker(
            worker_addr=worker_addr,
            model_path=args.model_name_or_path,
            model_names=args.model_names,
            conv_template="chatglm3",  # TODO 默认是chatglm3用于统一处理
        )
        if args.local_rank == "0":
            print("=======================================")
            print(f"{args.model_names[0]} 启动成功!")
            print("=======================================")
        uvicorn.run(app, host=host, port=port)
