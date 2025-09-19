import os
from typing import List
from loguru import logger
import numpy as np
from gpt_server.model_handler.pitch import pitch_flashtts

pitch_flashtts()
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from flashtts.server.utils.audio_writer import StreamingAudioWriter
import soundfile as sf
from voxcpm import VoxCPM

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class VoxCPMTTSWorker(ModelWorkerBase):
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
        self.model = VoxCPM.from_pretrained(model_path)
        logger.warning(f"模型：{model_names[0]}")

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
        speed = params["speed"]
        pitch = params["pitch"]
        sample_rate = 16 * 1000
        audio_writer = StreamingAudioWriter(
            format=response_format, sample_rate=sample_rate
        )
        generator = None
        wav = self.model.generate(
            text=text,
            prompt_wav_path=None,  # optional: path to a prompt speech for voice cloning
            prompt_text=None,  # optional: reference text
            cfg_value=2.0,  # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
            inference_timesteps=10,  # LocDiT inference timesteps, higher for better result, lower for fast speed
            normalize=True,  # enable external TN tool
            denoise=True,  # enable external Denoise tool
            retry_badcase=True,  # enable retrying mode for some bad cases (unstoppable)
            retry_badcase_max_times=3,  # maximum retrying times
            retry_badcase_ratio_threshold=6.0,  # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
        )

        # 分块处理（每块1024个样本）
        chunk_size = 1024
        for i in range(0, len(wav), chunk_size):
            chunk = wav[i : i + chunk_size]
            yield audio_writer.write_chunk(chunk.astype(np.float32))
        # 最终块处理
        end_chunk_data = audio_writer.write_chunk(finalize=True)
        yield end_chunk_data
     
        logger.debug(f"end_chunk_data 长度：{len(end_chunk_data)}")


if __name__ == "__main__":
    VoxCPMTTSWorker.run()
