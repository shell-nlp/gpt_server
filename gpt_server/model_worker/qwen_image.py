import os
from typing import List
import uuid
from loguru import logger
import shortuuid
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_worker.utils import pil_to_base64
import torch
from diffusers import DiffusionPipeline
from gpt_server.utils import STATIC_DIR

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
}

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]
import re


def contains_chinese(text):
    pattern = re.compile(r"[\u4e00-\u9fff]")
    return bool(pattern.search(text))


class QwenImageWorker(ModelWorkerBase):
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
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)

        logger.warning(f"模型：{model_names[0]}")

    async def get_image_output(self, params):
        prompt = params["prompt"]
        if contains_chinese(prompt):
            prompt += positive_magic["zh"]
        else:
            prompt += positive_magic["en"]
        response_format = params.get("response_format", "b64_json")
        image = self.pipe(
            prompt,
            negative_prompt=" ",
            height=height,
            width=width,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(self.device).manual_seed(0),
        ).images[0]
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
    QwenImageWorker.run()
