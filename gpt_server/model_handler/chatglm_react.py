from typing import Any, Dict, List, Tuple, Union
import json
import uuid

GLM4_TOOL_SUFFIX_PROMPT = "在调用上述函数时，请使用 Json 格式表示调用的参数。"

GLM4_TOOL_PROMPT = (
    "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持，"
    "{tool_text}"
)


def glm4_tool_formatter(tools: List[Dict[str, Any]]) -> str:
    tool_text = ""
    for tool in tools:
        tool = tool["function"]
        tool_name = tool["name"]
        tool_text += f"\n\n## {tool_name}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}\n{GLM4_TOOL_SUFFIX_PROMPT}"
    return GLM4_TOOL_PROMPT.format(tool_text=tool_text)


def glm4_tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:
    lines = content.strip().split("\n")
    if len(lines) != 2:
        return content
    tool_name = lines[0].strip()
    tool_input = lines[1].strip()
    try:
        json.loads(tool_input)
    except json.JSONDecodeError:
        return content
    tool_calls = [
        {
            "id": "call_{}".format(uuid.uuid4().hex),
            "function": {"name": tool_name, "arguments": tool_input},
        }
    ]

    return tool_calls


if __name__ == "__main__":
    import json

    tools_str = """[{'type': 'function', 'function': {'name': 'track', 'description': '追踪指定股票的实时价格', 'parameters': {'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码', 'type': 'integer'}}, 'required': ['symbol']}}}, {'type': 'function', 'function': {'name': 'text-to-speech', 'description': '将文本转换为语音', 'parameters': {'type': 'object', 'properties': {'text': {'description': '需要转换成语音的文本', 'type': 'string'}, 'voice': {'description': '要使用的语音类型（男声、女声等', 'default': '男声', 'type': 'string'}, 'speed': {'description': '语音的速度（快、中等、慢等', 'default': '中等', 'type': 'string'}}, 'required': ['text']}}}]"""
    tools_str = tools_str.replace("'", '"')
    tools = json.loads(tools_str)

    res = glm4_tool_formatter(tools=tools)
    print(res)
    print()
    out = 'multiply\n{"first_int": 8, "second_int": 9}'
    r = glm4_tool_extractor(out)
    print(r)
