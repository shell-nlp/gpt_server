import asyncio
import os
from typing import List
import base64
from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from io import BytesIO

from gpt_server.fast_tts.engine import AutoEngine
from gpt_server.fast_tts.server.utils.audio_writer import StreamingAudioWriter

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


async def generate_voice_stream(engine: AutoEngine, text: str, name: str = None):
    """
    流式音频合成
    """

    if engine.engine_name != "spark":
        raise ValueError("仅Spark-TTS支持`generate_voice_stream`功能.")
    stream_coro = None
    if name:
        stream_coro = engine.speak_stream_async(
            name=name,
            text=text,
            length_threshold=50,
            window_size=50,
        )
    else:
        stream_coro = engine.generate_voice_stream_async(
            text=text,
            gender="male",
            length_threshold=50,
            window_size=50,
        )
    async for chunk_data in stream_coro:
        yield chunk_data


os.environ["VLLM_USE_V1"] = "0"


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
            model_type="asr",
        )

        self.engine = AutoEngine(
            model_path=model_path,
            max_length=32768,
            llm_device="cuda:0",
            tokenizer_device="cuda:0",
            detokenizer_device="cuda:0",
            backend="vllm",
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

    async def generate_voice_stream(self, params):
        text = params["text"]
        response_format = params["response_format"]
        audio_writer = StreamingAudioWriter(
            format=response_format, sample_rate=self.engine.SAMPLE_RATE
        )
        async for chunk_data in generate_voice_stream(
            engine=self.engine, text=text, name="新闻联播女声"
        ):
            audio = audio_writer.write_chunk(chunk_data, finalize=False)
            yield audio
        yield audio_writer.write_chunk(finalize=True)


if __name__ == "__main__":
    SparkTTSWorker.run()
