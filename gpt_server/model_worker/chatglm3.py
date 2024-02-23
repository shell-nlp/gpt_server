import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
import torch
from gpt_server.model_handler.chatglm3 import conv2messages
from gpt_server.model_worker.base import ModelWorkerBase


class ChatGLM3Worker(ModelWorkerBase):
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

        self.stop_words_ids = [
            64795,
            64797,
            2,
        ]
        self.stop = [
            self.tokenizer.decode(skip_word) for skip_word in self.stop_words_ids
        ]

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
        input_ids.extend([self.tokenizer.get_command("<|assistant|>")])
        return self.tokenizer.batch_encode_plus(
            [input_ids], return_tensors="pt", is_split_into_words=True
        )

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        print("params", params)
        print("worker_id:", self.worker_id)
        try:
            # prompt = params["prompt"]
            # query, messages = conv2messages(prompt=prompt)
            # ----------------添加对工具的支持-----------------------------------
            # for item in messages:
            #     if item["role"] == "system":
            #         content = item["content"]
            #         if "tools:" in content:  # 说明使用chatglm3官方的指令
            #             print("chatglm3 使用工具!")
            #             idx = content.rfind("tools:")
            #             tools = content[idx + len("tools:") :]
            #             tools_obj = json.loads(tools)
            #             # -------------
            #             item["content"] = content[: idx + len("tools:")]
            #             item["tools"] = tools_obj
            # ----------------添加对工具的支持-----------------------------------
            messages = params["messages"]
            if messages[-1]["role"] == "user":
                last_message = messages.pop()
                query = last_message["content"]
                role = "user"  # 下一个角色是什么
            elif messages[-1]["role"] == "function":
                messages[-1]["role"] = "observation"
                query = ""
                role = "assistant"  # 下一个角色是什么
            input_ids = self.build_chat_input(query, history=messages, role=role)[
                "input_ids"
            ]
            print(self.tokenizer.decode(input_ids.tolist()[0]))
            # ---------------添加额外的参数------------------------
            params["stop"].extend(self.stop)
            params["stop_words_ids"] = self.stop_words_ids
            params["input_ids"] = input_ids
            # ---------------添加额外的参数------------------------
            async for response, usage in self.backend.stream_chat(
                query=query, params=params
            ):
                ret = {"text": response, "error_code": 0, "usage": usage}

                yield json.dumps(ret).encode() + b"\0"

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            print(e)
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def get_embeddings(self, params):
        return super().get_embeddings(params)


if __name__ == "__main__":
    ChatGLM3Worker.run()
