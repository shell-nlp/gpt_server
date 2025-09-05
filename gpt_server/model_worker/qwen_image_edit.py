import asyncio

import io
import os
from typing import List
import uuid
from loguru import logger
import shortuuid
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_worker.utils import (
    pil_to_base64,
    load_base64_or_url,
    bytesio2image,
)
from gpt_server.utils import STATIC_DIR
import torch
from diffusers import QwenImageEditPipeline

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class QwenImageEditWorker(ModelWorkerBase):
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
            model_type="image",
        )
        backend = os.environ["backend"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = QwenImageEditPipeline.from_pretrained(model_path)
        self.pipe.to(torch.bfloat16)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=None)
        logger.warning(f"模型：{model_names[0]}")

    async def get_image_output(self, params):
        prompt = params["prompt"]
        response_format = params.get("response_format", "b64_json")
        bytes_io = await load_base64_or_url(params["image"])
        image = bytesio2image(bytes_io)
        inputs = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": None,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }
        with torch.inference_mode():
            output = self.pipe(**inputs)
            image = output.images[0]

        result = {}
        if response_format == "b64_json":
            # Convert PIL image to base64
            base64 = pil_to_base64(pil_img=image)
            result = {
                "created": shortuuid.random(),
                "data": [{"b64_json": base64}],
                "usage": {
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
                },
            }
            return result
        elif response_format == "url":
            # 生成唯一文件名（避免冲突）
            file_name = str(uuid.uuid4()) + ".png"
            save_path = STATIC_DIR / file_name
            image.save(save_path, format="PNG")
            WORKER_PORT = os.environ["WORKER_PORT"]
            WORKER_HOST = os.environ["WORKER_HOST"]
            url = f"http://{WORKER_HOST}:{WORKER_PORT}/static/{file_name}"
            result = {
                "created": shortuuid.random(),
                "data": [{"url": url}],
                "usage": {
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
                },
            }
        return result


if __name__ == "__main__":
    QwenImageEditWorker.run()
