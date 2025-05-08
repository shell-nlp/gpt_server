import asyncio

import os
from typing import List
from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase

from flashtts.engine import AutoEngine
from flashtts.server.utils.audio_writer import StreamingAudioWriter

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


os.environ["VLLM_USE_V1"] = "0"
import httpx
from fastapi import HTTPException
import base64
import io


async def get_audio_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载参考音频")
        return response.content


async def load_base64_or_url(audio):
    # 根据 reference_audio 内容判断读取方式
    if audio.startswith("http://") or audio.startswith("https://"):
        audio_bytes = await get_audio_bytes_from_url(audio)
    else:
        try:
            audio_bytes = base64.b64decode(audio)
        except Exception as e:
            logger.warning("无效的 base64 音频数据: " + str(e))
            raise HTTPException(
                status_code=400, detail="无效的 base64 音频数据: " + str(e)
            )
    # 利用 BytesIO 包装字节数据，然后使用 soundfile 读取为 numpy 数组
    try:
        bytes_io = io.BytesIO(audio_bytes)
    except Exception as e:
        logger.warning("读取参考音频失败: " + str(e))
        raise HTTPException(status_code=400, detail="读取参考音频失败: " + str(e))
    return bytes_io


class SparkTTSWorker(ModelWorkerBase):
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
            model_type="tts",
        )
        backend = os.environ["backend"]
        self.engine = AutoEngine(
            model_path=model_path,
            max_length=32768,
            llm_device="auto",
            tokenizer_device="auto",
            detokenizer_device="auto",
            backend=backend,
            wav2vec_attn_implementation="sdpa",  # 使用flash attn加速wav2vec
            llm_gpu_memory_utilization=0.6,
            seed=0,
        )
        loop = asyncio.get_running_loop()
        # ------------- 添加声音 -------------
        loop.create_task(
            self.engine.add_speaker(
                "新闻联播女声",
                audio=os.path.join(
                    root_dir, "assets/audio_data/roles/新闻联播女声/女声.wav"
                ),
            )
        )
        logger.info(f"模型：{model_names[0]}")

    # 这个是模型主要的方法
    async def generate_voice_stream(self, params):
        if self.engine.engine_name != "spark":
            raise ValueError("仅Spark-TTS支持`generate_voice_stream`功能.")
        async for chunk_data in self.stream_async(params=params):
            yield chunk_data

    async def stream_async(self, params):
        text = params["text"]
        voice = params.get("voice", "新闻联播女声")
        response_format = params["response_format"]
        audio_writer = StreamingAudioWriter(
            format=response_format, sample_rate=self.engine.SAMPLE_RATE
        )
        generator = None
        if voice in self.engine.list_roles():
            generator = self.engine.speak_stream_async(
                name=voice,
                text=text,
                length_threshold=50,
                window_size=50,
            )
        else:  # clone
            reference_audio = await load_base64_or_url(voice)
            generator = self.engine.clone_voice_stream_async(
                text=text,
                reference_audio=reference_audio,
                length_threshold=50,
                window_size=50,
            )
        async for chunk_data in generator:
            audio = audio_writer.write_chunk(chunk_data, finalize=False)
            yield audio
        yield audio_writer.write_chunk(finalize=True)


if __name__ == "__main__":
    SparkTTSWorker.run()
