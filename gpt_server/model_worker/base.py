import asyncio
from typing import List
import json
from abc import ABC, abstractmethod
from fastapi import BackgroundTasks, Request, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.utils import (
    get_context_length,
)
from vllm.utils import random_uuid
from loguru import logger
import os
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoConfig,
)
import torch
import uuid
from gpt_server.utils import get_free_tcp_port

worker = None
app = FastAPI()


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
        self.USE_VLLM = os.getenv("USE_VLLM", 0)
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model_tokenizer(model_path)
        self.context_len = self.get_context_length()
        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.init_heart_beat()
        global worker
        if worker is None:
            worker = self
            print("worker 已赋值")

    def get_context_length(
        self,
    ):
        """ "支持的最大 token 长度"""
        if self.model is None:
            return 512
        self.model_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        return get_context_length(self.model_config)

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

    def load_model_tokenizer(self, model_path):
        """加载 模型 和 分词器 直接对 self.model 和 self.tokenizer 进行赋值"""
        if self.model_type == "embedding":
            return 1
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            encode_special_tokens=True,
        )
        if self.USE_VLLM:
            from gpt_server.model_backend.vllm_backend import VllmBackend

            logger.info("使用vllm 后端")
            self.backend = VllmBackend(model_path=self.model_path)
        else:
            from gpt_server.model_backend.hf_backend import HFBackend

            logger.info("使用hf 后端")
            MODEL_CLASS = self.get_model_class()
            self.model = MODEL_CLASS.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).half()

            self.model = self.model.eval()
            # 加载 HF 后端
            self.backend = HFBackend(tokenizer=self.tokenizer, model=self.model)

    @abstractmethod
    def generate_stream_gate(self, params):
        pass

    async def generate_gate(self, params):
        async for x in self.generate_stream_gate(params):
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
        model_names: List[str] = [""],
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

        parser.add_argument(
            "--model_name_or_path", type=str, default="model_name_or_path"
        )
        parser.add_argument(
            "--model_names", type=lambda s: s.split(","), default="model_names"
        )

        args = parser.parse_args()

        host = "localhost"
        port = get_free_tcp_port()
        worker_addr = f"http://{host}:{port}"

        worker = cls.get_worker(
            worker_addr=worker_addr,
            model_path=args.model_name_or_path,
            model_names=args.model_names,
            conv_template="chatglm3",  # TODO 默认是chatglm3用于统一处理
        )

        uvicorn.run(app, host=host, port=port)


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await worker.backend.engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    #
    if os.getenv("USE_VLLM", 0):
        background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    params.pop("prompt")
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    params.pop("prompt")
    output = await worker.generate_gate(params)
    release_worker_semaphore()
    if os.getenv("USE_VLLM", 0):
        await worker.backend.engine.abort(request_id)
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)
