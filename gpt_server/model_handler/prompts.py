from typing import Optional
from lmdeploy.model import MODELS, Qwen7BChat
import json


@MODELS.register_module(name="qwen2_5")
class Qwen2d5Chat(Qwen7BChat):
    """Chat template for Qwen2.5-Instruct series."""

    def __init__(
        self,
        system="<|im_start|>system\n",
        meta_instruction="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        eosys="<|im_end|>\n",
        user="<|im_start|>user\n",
        eoh="<|im_end|>\n",
        assistant="<|im_start|>assistant\n",
        eoa="<|im_end|>",
        separator="\n",
        tools="""\n\n# Tools\n\n您可以调用一个或多个function来协助处理用户查询。\n\n如果,你调用function,你必须要把function tag放在<tools></tools> XML 标签中间:\n<tools>""",
        eotools="""\n</tools>\n\n对于单个function的调用，返回一个包含function name和参数的 JSON 对象，并用 <tool_call></tool_call> XML 标签包裹,例如:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>""",
        stop_words=["<|im_end|>"],
        **kwargs,
    ):

        self.tools = tools
        self.eotools = (
            eotools
            + """\n如果需要同时调用多个function,则返回多个包含function name和参数的 JSON 对象，并用 <tool_call></tool_call> XML 标签包裹,例如:
<tool_call>
{"name": <function-name-1>, "arguments": <args-json-object>}
</tool_call>
<tool_call>
{"name": <function-name-2>, "arguments": <args-json-object>}
</tool_call>
"""
        )
        super().__init__(
            system=system,
            meta_instruction=meta_instruction,
            eosys=eosys,
            user=user,
            eoh=eoh,
            assistant=assistant,
            eoa=eoa,
            separator=separator,
            stop_words=stop_words,
            **kwargs,
        )

    def messages2prompt(self, messages, sequence_start=True, tools=None, **kwargs):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        box_map = dict(user=self.user, assistant=self.assistant, system=self.system)
        ret = ""
        tool_prompt = ""
        if tools is not None and len(tools) > 0:
            for tool in tools:
                tool_prompt += self.separator
                tool_prompt += f'{{"type": "function", "function": {json.dumps(tool, ensure_ascii=False)}}}'
            if len(messages) and messages[0]["role"] == "system":
                ret += f"{self.system}{messages[0]['content']}{self.tools}{tool_prompt}{self.eotools}{self.eosys}"
            else:
                ret += f"{self.system}{self.meta_instruction}{self.tools}{tool_prompt}{self.eotools}{self.eosys}"
        else:
            if self.meta_instruction is not None and sequence_start:
                if len(messages) and messages[0]["role"] == "system":
                    ret += f"{self.system}{messages[0]['content']}{self.eosys}"
                else:
                    ret += f"{self.system}{self.meta_instruction}{self.eosys}"

        for index, message in enumerate(messages):
            if (
                message["role"] == "user"
                or (message["role"] == "system" and index != 0)
                or (
                    message["role"] == "assistant" and message.get("tool_calls") is None
                )
            ):
                ret += f"{box_map[message['role']]}{message['content']}{self.eosys}"
            elif message["role"] == "assistant":
                ret += f"<|im_start|>assistant"
                if message.get("content") is not None:
                    ret += f"{self.separator}{message['content']}"

                if message.get("tool_calls") is not None:
                    tool_calls = message["tool_calls"]
                    for tool_call in tool_calls:
                        if tool_call.get("function") is not None:
                            tool_call = tool_call["function"]
                        if isinstance(tool_call["arguments"], str):
                            tool_call["arguments"] = json.loads(tool_call["arguments"])
                        ret += f'{self.separator}<tool_call>{self.separator}{{"name": "{tool_call["name"]}", "arguments": {json.dumps(tool_call["arguments"], ensure_ascii=False)}}}{self.separator}</tool_call>'
                ret += self.eosys
            if message["role"] == "tool":
                if index == 0 or messages[index - 1]["role"] != "tool":
                    ret += f"<|im_start|>user"
                ret += f"{self.separator}<tool_response>{self.separator}{message['content']}{self.separator}</tool_response>"
                if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
                    ret += f"{self.eoh}"
        ret += f"{self.assistant}"
        return ret

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        lower_path = model_path.lower()
        if "qwen2.5" in lower_path or "qwen2_5" in lower_path:
            return "qwen2d5"
