from typing import Any, Dict, List, Tuple, Union, Optional
import json
import uuid

from gpt_server.model_handler.react.prompt import (
    GLM4_TOOL_PROMPT,
    TOOL_SUFFIX_PROMPT,
)


def system_tool_formatter(
    tools: List[Dict[str, Any]], tool_choice_info: Optional[dict] = None
) -> str:
    tool_text = "\n"
    tool_names = []
    for tool in tools:
        tool = tool["function"]
        tool_name = tool["name"]
        tool_text += f"## {tool_name}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}\n{TOOL_SUFFIX_PROMPT}\n\n"
        tool_names.append(tool_name)
    return GLM4_TOOL_PROMPT.format(
        tool_text=tool_text, tool_names=", ".join(tool_names)
    ).strip()


def system_tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:
    i = content.rfind("Action:")
    j = content.rfind("Action Input:")
    tool_name = content[i + len("Action:") : j].strip().strip(".")
    tool_input = content[j + len("Action Input:") :].strip()
    try:
        tool_input_obj = json.loads(tool_input)
    except json.JSONDecodeError:
        return content
    tool_calls = []
    tool_call = {
        "index": 0,
        "id": "call_{}".format(uuid.uuid4().hex),
        "function": {"name": tool_name, "arguments": tool_input},
    }
    tool_calls.append(tool_call)

    return tool_calls


if __name__ == "__main__":
    import json

    tools_str = """[{'type': 'function', 'function': {'name': 'track', 'description': '追踪指定股票的实时价格', 'parameters': {'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码', 'type': 'integer'}}, 'required': ['symbol']}}}, {'type': 'function', 'function': {'name': 'text-to-speech', 'description': '将文本转换为语音', 'parameters': {'type': 'object', 'properties': {'text': {'description': '需要转换成语音的文本', 'type': 'string'}, 'voice': {'description': '要使用的语音类型（男声、女声等', 'default': '男声', 'type': 'string'}, 'speed': {'description': '语音的速度（快、中等、慢等', 'default': '中等', 'type': 'string'}}, 'required': ['text']}}}]"""
    tools_str = tools_str.replace("'", '"')
    tools = json.loads(tools_str)

    res = system_tool_formatter(tools=tools)
    print(res)
    print()
    out = 'multiply\n{"first_int": 8, "second_int": 9}'
    r = system_tool_extractor(out)
    print(r)
