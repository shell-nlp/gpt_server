import json
import re
from typing import List, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field
import shortuuid
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    FunctionCall,
)

from vllm.tool_parsers import ToolParser, ToolParserManager


class ToolCall(BaseModel):
    """Tool call response."""

    index: Optional[int] = None
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    type: Literal["function"] = "function"
    function: FunctionCall


class ExtractedToolCallInformation(BaseModel):
    # modified from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/openai/protocol.py#L1199
    # indicate if tools were called
    tools_called: bool
    # extracted tool calls
    tool_calls: List[ToolCall]
    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


@ToolParserManager.register_module(["qwen2_5"])
class Qwen2d5ToolParser(ToolParser):
    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.position = 0
        self.tool_start_token = "<tool_call>"
        self.tool_end_token = "</tool_call>"
        self.pattern = r"<tool_call>(.*?)</tool_call>"

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output
        if self.tool_start_token in text and self.tool_end_token in text:
            logger.debug("tool_parse tool_start_token 在 text")
            # get tool_call in text
            match_result_list = re.findall(self.pattern, text, re.DOTALL)
            tool_calls = []
            index = -1
            for match_result in match_result_list:
                index += 1
                action = json.loads(match_result)
                name = action["name"]
                try:
                    arguments = json.dumps(action["arguments"], ensure_ascii=False)
                except KeyError:
                    arguments = json.dumps(action["parameters"], ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        index=index,
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                )

            # get text outside of tags
            if not text.startswith("<tool_call>"):
                text = text[: text.find("<tool_call>")]
            elif not text.endswith("</tool_call>"):
                text = text[text.rfind("</tool_call>") + len("</tool_call>") :]
            else:
                text = ""
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=text if len(text) > 0 else "",
            )
        elif self.tool_start_token in text or self.tool_end_token in text:
            # 如果 tool_start_token 不在 text 但是 tool_end_token 在text
            logger.debug("tool_parse tool_start_token 不在 text")
            pattern = r"\{[^{}]*\{[^{}]*\}[^{}]*\}|{[^{}]*}"
            match_result_list = re.findall(pattern, text, re.DOTALL)
            tool_calls = []
            tools_called = False
            index = -1
            # parameters
            for match_result in match_result_list:
                index += 1
                action = json.loads(match_result)
                name = action["name"]
                try:
                    arguments = json.dumps(action["arguments"], ensure_ascii=False)
                except KeyError:
                    arguments = json.dumps(action["parameters"], ensure_ascii=False)

                tool_calls.append(
                    ToolCall(
                        function=FunctionCall(name=name, arguments=arguments),
                        index=index,
                    )
                )
                tools_called = True
                # get text outside of tags

            return ExtractedToolCallInformation(
                tools_called=tools_called,
                tool_calls=tool_calls,
                content=text if len(text) > 0 else "",
            )
        logger.debug("tool_parse 无结果")
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=text
        )


def tool_parser(full_text: str, tool_parser_: ToolParser, tools, ret):
    try:
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": full_text}], tools=tools
        )
        tool_call_info = tool_parser_.extract_tool_calls(
            model_output=full_text, request=request
        )
        tools_called = tool_call_info.tools_called
        _, tool_calls_ = tool_call_info.content, tool_call_info.tool_calls
        tool_calls = []
        for index, i in enumerate(tool_calls_):
            tool_call = i.model_dump()
            if "index" not in tool_call:
                tool_call["index"] = index
            tool_calls.append(tool_call)

        # -----------------------------------
        ret["text"] = ""
        ret["tool_calls"] = tool_calls
        ret["finish_reason"] = (
            "tool_calls" if tools and tools_called else ret.get("finish_reason", "stop")
        )
        if tools:
            logger.info(
                f" 工具解析{'成功' if tools_called else '失败'}, tool_calls: {tool_calls}"
            )
        if not tools_called:
            return None
        return json.dumps(ret).encode() + b"\0"
    except Exception as e:
        logger.warning(f"Error in tool_parser: {e}")
        import traceback

        traceback.print_exc()
        return None


import json
import logging
from typing import Dict, List, Any, Optional


class ToolCallStreamProcessor:
    """
    处理流式tool_calls，只接收tool_calls部分数据
    """

    def __init__(self):
        # 存储所有工具调用的累积数据，按index索引
        self.tool_calls: Dict[int, Dict[str, Any]] = {}

    def process_chunk(self, tool_calls_data: List[Dict]) -> Optional[List[Dict]]:
        """
        处理tool_calls数据
        参数: tool_calls_data - 从delta中提取的tool_calls列表
        返回: 如果检测到完成则返回完整的工具调用，否则返回None
        """
        if not tool_calls_data:
            return None

        for tool_call in tool_calls_data:
            index = tool_call.get("index", 0)

            # 初始化新工具调用
            if index not in self.tool_calls:
                self.tool_calls[index] = {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                }

            current = self.tool_calls[index]

            # 更新ID（只在第一个chunk中出现）
            if tool_call.get("id"):
                current["id"] = tool_call["id"]

            # 更新函数名（只在第一个chunk中出现）
            function_data = tool_call.get("function", {})
            if function_data.get("name"):
                current["function"]["name"] = function_data["name"]

            # 累积参数字符串
            if function_data.get("arguments"):
                current["function"]["arguments"] += function_data["arguments"]

        return None

    def get_completed_tool_calls(self) -> Optional[List[Dict]]:
        """
        获取所有完整的工具调用，并解析arguments JSON
        通常在收到finish_reason='tool_calls'后调用
        """
        if not self.tool_calls:
            return None

        completed_calls = []

        for index in sorted(self.tool_calls.keys()):
            call_data = self.tool_calls[index]

            # 检查是否完整
            if not call_data["id"] or not call_data["function"]["name"]:
                logging.warning(f"工具调用 {index} 不完整，跳过")
                continue

            # 解析arguments JSON
            args_str = call_data["function"]["arguments"]

            completed_calls.append(
                {
                    "id": call_data["id"],
                    "type": call_data["type"],
                    "function": {
                        "name": call_data["function"]["name"],
                        "arguments": args_str,
                    },
                }
            )

        return completed_calls if completed_calls else None

    def reset(self):
        """重置处理器"""
        self.tool_calls = {}


if __name__ == "__main__":
    from transformers import AutoTokenizer

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
        }
    ]
    glm_full_text = """Action: get_weather
Action Input: {"location": "Nanjing", "unit": "celsius"}"""
    qwen_full_text = """<tool_call>{"name": "get_weather", "arguments": {"location": "Nanjing", "unit": "celsius"}}</tool_call>"""
    qwen3coder_text = """
<tool_call>
<function=get_weather>
<parameter=location>
南京
</parameter>
<parameter=unit>
celsius
</parameter>
</function>
</tool_call>
"""
    tokenizer = AutoTokenizer.from_pretrained("/home/dev/model/Qwen/Qwen3___5-35B-A3B/")
    tool_parser_ = ToolParserManager.get_tool_parser("qwen2_5")(tokenizer)
    tool_parser(
        full_text=qwen_full_text, tool_parser_=tool_parser_, tools=tools, ret={}
    )
