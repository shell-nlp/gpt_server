import os
from typing import List
import base64
from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO

class FunASRWorker(ModelWorkerBase):
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
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
            device = "cpu"
        else:
            device = "cuda"
        logger.info(f"使用{device}加载...")
        self.model = AutoModel(
            model=model_path,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda",
        )
        logger.info(f"模型：{model_names[0]}")

    async def transcription(self, params):
        file_input = base64.b64decode(params["file"])  # Base64 → bytes
        file_input = BytesIO(file_input)
        ret = {}
        res = self.model.generate(
            input=file_input,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        ret["text"] = text
        return ret


if __name__ == "__main__":
    FunASRWorker.run()
