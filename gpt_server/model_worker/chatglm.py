import asyncio
import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
import torch
from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase
from gpt_server.model_handler.prompts import MODELS
from gpt_server.model_handler.tool_parser import tool_parser, ToolParserManager


def pop_matching_tool(tools, tool_choice):
    # 获取目标function名称
    target_name = tool_choice["function"]["name"]

    # 遍历tools列表，查找匹配项
    for index, tool in enumerate(tools):
        if tool["function"]["name"] == target_name:
            return [tools.pop(index)]

    # 未找到时返回None
    return None


class ChatGLMWorker(ModelWorkerBase):
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
            model_type="AutoModel",
            multimodal=False,
        )
        self.chat_template = MODELS.module_dict["glm4"]()
        self.tool_parser = ToolParserManager.module_dict["glm"](
            tokenizer=self.tokenizer
        )
        self.stop_words_ids = []

        self.stop = [
            self.tokenizer.decode(skip_word) for skip_word in self.stop_words_ids
        ]
        logger.warning(f"{model_names[0]} 停用词: {self.stop}")

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        try:
            messages = params.get("messages", [])
            tools = params.get("tools", None)
            tool_choice = params.get("tool_choice", "none")
            if tool_choice == "none":
                tools = None
            elif tool_choice == "auto" or tool_choice == "required":
                pass
            elif isinstance(tool_choice, dict):
                tools = pop_matching_tool(tools=tools, tool_choice=tool_choice)
            if not self.vision_config:
                if isinstance(messages, list):
                    text = await asyncio.to_thread(
                        self.chat_template.messages2prompt, messages, True, tools
                    )
                elif isinstance(messages, str):
                    text = messages
                    # input_ids = self.tokenizer([text], return_tensors="pt").input_ids

                # text = self.tokenizer.decode(input_ids.tolist()[0])
                params["prompt"] = text
                # params["input_ids"] = input_ids

            # ---------------添加额外的参数------------------------
            params["messages"] = messages
            params["stop"].extend(self.stop)
            params["stop_words_ids"] = self.stop_words_ids
            # ---------------添加额外的参数------------------------
            full_text = ""
            ret = {}
            async for ret in self.backend.stream_chat(params=params):
                full_text += ret["text"]

                yield json.dumps(ret).encode() + b"\0"
            # ------ add tool_calls ------
            yield tool_parser(
                full_text=full_text, tool_parser=self.tool_parser, tools=tools, ret=ret
            )
            # ------ add tool_calls ------
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            logger.info(e)
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"


if __name__ == "__main__":
    ChatGLMWorker.run()
