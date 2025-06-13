import asyncio

import io
import os
from typing import List
import uuid
from loguru import logger
import shortuuid
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_worker.utils import pil_to_base64
from gpt_server.utils import STATIC_DIR
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class WanWorker(ModelWorkerBase):
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
        vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32
        )
        self.pipe = WanPipeline.from_pretrained(
            model_path, vae=vae, torch_dtype=torch.bfloat16
        ).to(self.device)
        logger.warning(f"模型：{model_names[0]}")

    async def get_image_output(self, params):
        prompt = params["prompt"]
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=5.0,
        ).frames[0]

        # 生成唯一文件名（避免冲突）
        file_name = str(uuid.uuid4()) + ".mp4"
        save_path = STATIC_DIR / file_name
        export_to_video(output, save_path, fps=15)
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
    WanWorker.run()
