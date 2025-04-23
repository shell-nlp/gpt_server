import asyncio
from typing import List
import json
from abc import ABC, abstractmethod
from fastapi import BackgroundTasks, Request, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastchat.utils import SEQUENCE_LENGTH_KEYS
from loguru import logger
import os
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoConfig,
)
import uuid
from gpt_server.utils import get_free_tcp_port
from gpt_server.model_worker.base.base_model_worker import BaseModelWorker

worker = None
app = FastAPI()


def get_context_length_(config):
    """Get the context length of a model from a huggingface model config."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        try:
            rope_scaling_factor = config.rope_scaling["factor"]
        except KeyError:
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


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
        multimodal: bool = False,
    ):
        is_vision = False
        if model_type != "asr":
            try:
                self.model_config = AutoConfig.from_pretrained(
                    model_path, trust_remote_code=True
                )
            except ValueError as e:
                logger.warning(e)
                self.model_config = {}
            # logger.info(f"模型配置：{self.model_config}")
            self.vision_config = getattr(self.model_config, "vision_config", None)
            is_vision = self.vision_config is not None
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
            multimodal=multimodal or is_vision,
        )
        os.environ["WORKER_NAME"] = self.__class__.__name__
        self.worker_name = self.__class__.__name__
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.backend = None
        self.tokenizer = None
        self.load_model_tokenizer(model_path)
        self.context_len = self.get_context_length()
        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        self.init_heart_beat()
        global worker
        if worker is None:
            worker = self
            logger.info("worker 已赋值")

    def get_context_length(
        self,
    ):
        """ "支持的最大 token 长度"""
        if self.model is None and self.backend is None:
            return 512
        return get_context_length_(self.model_config)

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
        if self.model_type == "embedding" or self.model_type == "asr":
            return 1
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            encode_special_tokens=True,
        )
        if os.getenv("backend") == "vllm":
            from gpt_server.model_backend.vllm_backend import VllmBackend

            logger.info(f"{self.worker_name} 使用 vllm 后端")
            self.backend = VllmBackend(
                model_path=self.model_path, tokenizer=self.tokenizer
            )
        elif "sglang" in os.getenv("backend"):
            from gpt_server.model_backend.sglang_backend import SGLangBackend

            logger.info(f"{self.worker_name} 使用 SGLang 后端")
            self.backend = SGLangBackend(model_path=self.model_path)
        elif "lmdeploy" in os.getenv("backend"):
            from gpt_server.model_backend.lmdeploy_backend import LMDeployBackend

            logger.info(f"{self.worker_name} 使用 LMDeploy 后端")
            self.backend = LMDeployBackend(model_path=self.model_path)

        elif os.getenv("backend") == "hf":
            from gpt_server.model_backend.hf_backend import HFBackend

            logger.info(f"{self.worker_name} 使用 hf 后端")
            MODEL_CLASS = self.get_model_class()
            self.model = MODEL_CLASS.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )

            self.model = self.model.eval()
            # 加载 HF 后端
            self.backend = HFBackend(tokenizer=self.tokenizer, model=self.model)
        logger.info("load_model_tokenizer 完成")

    async def generate_gate(self, params):
        full_text = ""
        async for ret in self.generate_stream_gate(params):
            full_text += json.loads(ret[:-1].decode()).get("text", "")
        ret = json.loads(ret[:-1].decode())
        ret["text"] = full_text
        return ret

    @classmethod
    def get_worker(
        cls,
        model_path: str,
        worker_addr: str,
        controller_addr: str = "http://localhost:21001",
        worker_id: str = str(uuid.uuid4())[:8],
        model_names: List[str] = [""],
        limit_worker_concurrency: int = 10000,
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
        parser.add_argument("--num_gpus", type=int, default=1)
        parser.add_argument("--backend", type=str, default="hf")

        parser.add_argument(
            "--model_name_or_path", type=str, default="model_name_or_path"
        )
        parser.add_argument(
            "--model_names", type=lambda s: s.split(","), default="model_names"
        )
        parser.add_argument("--lora", type=str, default=None)
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument(
            "--controller_address", type=str, default="http://localhost:21001"
        )
        parser.add_argument("--enable_prefix_caching", type=str, default="False")
        parser.add_argument("--dtype", type=str, default="auto")
        parser.add_argument("--max_model_len", type=str, default=None)
        parser.add_argument("--gpu_memory_utilization", type=str, default="0.8")
        # kv_cache_quant_policy
        parser.add_argument("--kv_cache_quant_policy", type=str, default="0")
        # vad_model
        parser.add_argument("--vad_model", type=str, default="")
        args = parser.parse_args()
        os.environ["num_gpus"] = str(args.num_gpus)
        if args.backend == "vllm":
            os.environ["backend"] = "vllm"
        elif args.backend == "hf":
            os.environ["backend"] = "hf"
        elif args.backend == "lmdeploy-pytorch":
            os.environ["backend"] = "lmdeploy-pytorch"
        elif args.backend == "lmdeploy-turbomind":
            os.environ["backend"] = "lmdeploy-turbomind"
        elif args.backend == "sglang":
            os.environ["backend"] = "sglang"

        if args.lora:
            os.environ["lora"] = args.lora
        if args.max_model_len:
            os.environ["max_model_len"] = args.max_model_len
        if args.vad_model:
            os.environ["vad_model"] = args.vad_model

        os.environ["enable_prefix_caching"] = args.enable_prefix_caching
        os.environ["gpu_memory_utilization"] = args.gpu_memory_utilization
        os.environ["kv_cache_quant_policy"] = args.kv_cache_quant_policy
        os.environ["dtype"] = args.dtype

        host = args.host
        controller_address = args.controller_address

        port = get_free_tcp_port()
        worker_addr = f"http://{host}:{port}"

        @app.on_event("startup")
        async def startup():
            global worker

            worker = cls.get_worker(
                worker_addr=worker_addr,
                model_path=args.model_name_or_path,
                model_names=args.model_names,
                conv_template="chatglm3",  # TODO 默认是chatglm3用于统一处理
                controller_addr=controller_address,
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
    if os.getenv("backend") == "vllm":
        background_tasks.add_task(abort_request)
    return background_tasks


request_id = 0


def gen_request_id():
    global request_id
    request_id += 1
    return str(request_id)


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = gen_request_id()
    params["request_id"] = request_id
    params["request"] = request
    params.pop("prompt")
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate_voice_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = gen_request_id()
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_voice_stream(params)
    background_tasks = create_background_tasks(request_id)
    response_format = params["response_format"]
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(response_format, f"audio/{response_format}")
    return StreamingResponse(
        generator,
        background=background_tasks,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{response_format}",
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = gen_request_id()
    params["request_id"] = request_id
    params["request"] = request
    params.pop("prompt")
    output = await worker.generate_gate(params)
    release_worker_semaphore()
    if os.getenv("backend") == "vllm":
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
    embedding = await worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)


@app.post("/worker_get_classify")
async def api_get_classify(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    outputs = await worker.classify(params)
    release_worker_semaphore()
    return JSONResponse(content=outputs)


@app.post("/worker_get_transcription")
async def api_get_transcription(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    outputs = await worker.transcription(params)
    release_worker_semaphore()
    return JSONResponse(content=outputs)
