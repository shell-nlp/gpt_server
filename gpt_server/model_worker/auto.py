import json
import traceback
from typing import List

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from loguru import logger
import torch
from vllm.tool_parsers import ToolParserManager

from gpt_server.model_handler.tool_parser import tool_parser
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_worker.utils import guess_tool_parser_by_model
from gpt_server.settings import get_model_config


class AutoWorker(ModelWorkerBase):
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
            model_type="AutoModelForCausalLM",
        )

        self.stop_words_ids = []

        self.stop = [
            self.tokenizer.decode(skip_word) for skip_word in self.stop_words_ids
        ]
        tool_parser_name = guess_tool_parser_by_model(model_path)
        model_config = get_model_config()

        # from https://github.com/xorbitsai/inference/blob/c70ea74fa820a613f8d577047ef1818da20a96b3/xinference/model/llm/llm_family_modelscope.json
        self.tool_parser = ToolParserManager.get_tool_parser(tool_parser_name)(
            self.tokenizer
        )
        logger.warning(
            f"已启动模型: {model_names[0]} |  工具解析器: {tool_parser_name} | 推理解析器: {model_config.reasoning_parser}"
        )

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        try:
            tools = params.get("tools", None)
            api_type = params.get("api_type", "chat")
            full_text = ""
            ret = {}
            if api_type == "chat":
                async for ret in self.backend.stream_chat(params=params):
                    full_text += ret.get("text", "")
                    yield json.dumps(ret).encode() + b"\0"
                # ------ add tool_calls ------
                yield tool_parser(
                    full_text=full_text,
                    tool_parser_=self.tool_parser,
                    tools=tools,
                    ret=ret,
                )
                # ------ add tool_calls ------
            else:
                async for ret in self.backend.stream_chat(params=params):
                    yield ret.encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            traceback.print_exc()
            logger.info(e)
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"


if __name__ == "__main__":
    AutoWorker.run()
