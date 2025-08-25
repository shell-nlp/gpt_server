from typing import Optional
from lmdeploy.model import MODELS, Qwen7BChat, ChatGLM3, get_text
import json


@MODELS.register_module(name="glm4", force=True)
class Glm4Chat(ChatGLM3):
    """Chat template of glm-4 model."""

    def __init__(
        self,
        system="<|system|>\n",
        user="<|user|>\n",
        assistant="<|assistant|>\n",
        separator="\n",
        tools="""\n\n你可以使用以下工具提供适当的答复和支持。\n\n# 可用工具\n\n在<tools></tools> XML标签中提供了function的签名(即函数的结构信息):\n<tools>""",
        eotools="""\n</tools>
## 如果使用工具，你可以在回复中插入零次、一次或多次以下命令以调用工具：

Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action

形如：
Action: get_result
Action Input: {{"a": "1","b": "2"}}

如果回答问题已经不需要再继续使用工具，则不需要再使用Action、Action Input格式，可直接回答。 
""",
        stop_words=["<|user|>", "<|endoftext|>", "<|observation|>"],
        meta_instruction="你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。",
        **kwargs,
    ):
        super().__init__(
            system=system,
            user=user,
            assistant=assistant,
            stop_words=stop_words,
            separator=separator,
            meta_instruction=meta_instruction,
            **kwargs,
        )
        self.start = "[gMASK]<sop>"
        self.tools = tools
        self.eotools = eotools

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if "glm-4" in path:
            return "glm4"

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
        return self.start + self.messages2prompt_base(
            messages, sequence_start, tools=tools, **kwargs
        )

    def messages2prompt_base(self, messages, sequence_start=True, tools=None, **kwargs):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """

        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        box_map = dict(
            user=self.user, assistant=self.assistant, system=self.system, tool=self.tool
        )
        eox_map = dict(
            user=self.eoh,
            assistant=self.eoa + self.separator,
            system=self.eosys,
            tool=self.eotool,
        )
        ret = ""
        if self.meta_instruction is not None and sequence_start:
            if len(messages) and messages[0]["role"] != "system":
                ret += f"{self.system}{self.meta_instruction}{self.eosys}"
        tool_prompt = ""
        if tools is not None and len(tools) > 0:
            tool_names = []
            for tool in tools:
                tool_names.append(tool["function"]["name"])
            tool_names = ",".join(tool_names)
            eotools = self.eotools.format(tool_names=tool_names)
            for tool in tools:
                tool_prompt += self.separator
                tool_prompt += f'{{"type": "function", "function": {json.dumps(tool, ensure_ascii=False)}}}'
            if len(messages) and messages[0]["role"] == "system":
                ret += f"{self.system}{messages[0]['content']}{self.tools}{tool_prompt}{eotools}{self.eosys}"
                messages.pop(0)
            else:
                ret += f"{self.system}{self.meta_instruction}{self.tools}{tool_prompt}{eotools}{self.eosys}"

        for message in messages:
            role = message["role"]
            content = get_text(message["content"])
            ret += f"{box_map[role]}{content}{eox_map[role]}"
        if (
            len(messages)
            and messages[-1]["role"] == "assistant"
            and len(eox_map["assistant"]) > 0
        ):
            return ret[: -len(eox_map["assistant"])]  # prefix of response
        ret += f"{self.assistant}"
        return ret


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
        tools="""\n\n# Tools\n\n您可以调用一个或多个function来协助处理用户查询。\n\n在<tools></tools> XML标签中提供了function的签名(即函数的结构信息):\n<tools>""",
        eotools="""\n</tools>\n\n对于单个function的调用，返回一个包含function name和参数的 JSON 对象，并用 <tool_call></tool_call> XML 标签包裹,形如:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>""",
        stop_words=["<|im_end|>"],
        **kwargs,
    ):

        self.tools = tools
        self.eotools = (
            eotools
            + """\n如果需要同时调用多个function,则返回多个包含function name和参数的 JSON 对象，并用 <tool_call></tool_call> XML 标签包裹,形如:
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

    def messages2prompt(
        self, messages, sequence_start=True, tools=None, enable_thinking=None, **kwargs
    ):
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
                name = message.get("name", "")
                ret += f"<|im_start|>assistant name: {name}"
                if (
                    message.get("content")
                    is not None
                    # and message.get("tool_calls") is None
                ):  # 是否添加and message.get("tool_calls") is None 来去掉带tool的 content内容
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
        if enable_thinking is False:
            ret += "<think>\n\n</think>\n\n"
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


if __name__ == "__main__":
    chat_template = MODELS.module_dict["qwen2_5"]()
    messages = [
        {"role": "system", "content": "我的Qwen "},
        {"role": "user", "content": "你是谁 "},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g., 'San Francisco, CA'",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather2",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g., 'San Francisco, CA'",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
    ]
    # tools = None
    prompt = chat_template.messages2prompt(messages, True, tools)
    print(prompt)
