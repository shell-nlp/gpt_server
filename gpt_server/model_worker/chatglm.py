import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
import torch
from loguru import logger
from gpt_server.model_worker.base import ModelWorkerBase
from gpt_server.model_handler.react.chatglm_react import glm4_tool_extractor
from gpt_server.model_handler.utils import add_tools2messages
from transformers import AutoConfig


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
        )

        self.stop = ["<|user|>", "<|observation|>", "<|endoftext|>"]
        # 拓展额外的stop
        self.stop.extend(["Observation:"])
        self.stop_words_ids = []
        for i in self.stop:
            try:
                self.stop_words_ids.append(self.tokenizer.convert_tokens_to_ids(i))
            except Exception as e:
                pass
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.vision_config = getattr(config, "vision_config", None)
        logger.info(f"chatglm停用词: {self.stop}")

    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = (
                    content
                    + "\n"
                    + json.dumps(item["tools"], indent=4, ensure_ascii=False)
                )
            input_ids.extend(
                self.tokenizer.build_single_message(
                    item["role"], item.get("metadata", ""), content
                )
            )
        if role == "user":
            input_ids.extend(self.tokenizer.build_single_message(role, "", query))
        input_ids.extend([self.tokenizer.convert_tokens_to_ids("<|assistant|>")])
        return self.tokenizer.batch_encode_plus(
            [input_ids], return_tensors="pt", is_split_into_words=True
        )

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        logger.info(f"params {params}")
        logger.info(f"worker_id: {self.worker_id}")
        try:
            # ----------------添加对工具的支持-----------------------------------
            messages = add_tools2messages(params=params, model_adapter="chatglm4")
            if not self.vision_config:
                if isinstance(messages, list):
                    task = "chat"
                    for msg in messages:
                        if msg["role"] == "function" or msg["role"] == "tool":
                            msg["role"] = "observation"

                    if messages[-1]["role"] == "user":
                        last_message = messages.pop()
                        query = last_message["content"]
                        role = "user"  # 下一个角色是什么
                    elif messages[-1]["role"] == "observation":
                        query = ""
                        role = "assistant"  # 下一个角色是什么
                    elif messages[-1]["role"] == "assistant":
                        query = ""
                        role = "user"
                    input_ids = self.build_chat_input(
                        query, history=messages, role=role
                    )["input_ids"]
                elif isinstance(messages, str):
                    task = "completion"
                    text = messages
                    input_ids = self.tokenizer([text], return_tensors="pt").input_ids

                text = self.tokenizer.decode(input_ids.tolist()[0])
                logger.info(text)
                params["prompt"] = text
                params["input_ids"] = input_ids
            # ---------------添加额外的参数------------------------
            params["messages"] = messages
            params["stop"].extend(self.stop)
            params["stop_words_ids"] = self.stop_words_ids
            # ---------------添加额外的参数------------------------
            async for ret in self.backend.stream_chat(params=params):
                response = ret["text"]

                yield json.dumps(ret).encode() + b"\0"
            # ------ add tool_calls ------
            tool_calls = glm4_tool_extractor(response)
            if params.get("tools", False) and isinstance(
                tool_calls, list
            ):  # 如果传入tools
                logger.debug(f"工具解析成功, tool_calls: {tool_calls}")
                ret["tool_calls"] = tool_calls
                yield json.dumps(ret).encode() + b"\0"
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

    def get_embeddings(self, params):
        return super().get_embeddings(params)


if __name__ == "__main__":
    ChatGLMWorker.run()
