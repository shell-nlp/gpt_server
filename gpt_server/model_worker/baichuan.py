import json
from typing import List
from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
import torch
from loguru import logger
from gpt_server.model_worker.base.model_worker_base import ModelWorkerBase


def build_chat_input(tokenizer, messages: List[dict], max_new_tokens: int = 0):
    user_token_id = 195
    assistant_token_id = 196

    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or 2048
    max_input_tokens = 4096 - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(user_token_id)
            else:
                round_tokens.append(assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if (
            len(history_tokens) == 0
            or len(history_tokens) + len(round_tokens) <= max_history_tokens
        ):
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens])


class BaiChuanWorker(ModelWorkerBase):
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
        self.stop_words_ids = [
            2,  # </s>
        ]
        self.stop = [
            self.tokenizer.decode(skip_word) for skip_word in self.stop_words_ids
        ]
        logger.warning(f"{model_names[0]} 停用词: {self.stop}")

    async def generate_stream_gate(self, params):
        self.call_ct += 1
        try:
            messages = params["messages"]
            if isinstance(messages, list):
                task = "chat"
            elif isinstance(messages, str):
                task = "completion"
            if task == "chat":
                input_ids = build_chat_input(
                    tokenizer=self.tokenizer, messages=messages
                )
                text = self.tokenizer.decode(input_ids.tolist()[0])
            elif task == "completion":
                text = messages
                input_ids = self.tokenizer([text], return_tensors="pt").input_ids

            params["messages"] = messages
            params["prompt"] = text
            params["stop"].extend(self.stop)
            params["stop_words_ids"] = self.stop_words_ids
            params["input_ids"] = input_ids

            async for ret in self.backend.stream_chat(params=params):
                response = ret["text"]

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


if __name__ == "__main__":
    BaiChuanWorker.run()
