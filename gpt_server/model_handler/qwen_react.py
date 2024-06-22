import re
from typing import Any, Dict, List, Tuple, Union
import json
import uuid

# default
TOOL_SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question:"""


def qwen_tool_formatter(tools: List[Dict[str, Any]]) -> str:
    tool_names = []
    param_text_list = []
    for tool in tools:
        tool = tool["function"]
        param_text = """{tool_name}: Call this tool to interact with the {tool_name} API. What is the {tool_name} API useful for? {description} Parameters: {parameters} Format the arguments as a JSON object."""
        parameters = []
        for name, param in tool["parameters"]["properties"].items():
            parameters.append(
                {
                    "name": name,
                    "description": param.get("description", ""),
                    "required": (
                        True if name in tool["parameters"]["required"] else False
                    ),
                    "schema": {"type": param["type"]},
                }
            )
        param_text_str = param_text.format(
            tool_name=tool["name"],
            description=tool["description"],
            parameters=parameters,
        )
        param_text_list.append(param_text_str)

        tool_names.append(tool["name"])

    tool_text = "\n\n".join(param_text_list).strip()
    return TOOL_SYSTEM_PROMPT.format(
        tool_text=tool_text,
        tool_names=", ".join(tool_names),
    )


def qwen_tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:

    i = content.rfind("Action:")
    j = content.rfind("Action Input:")
    tool_name = content[i + len("Action:") : j].strip().strip(".")
    tool_input = content[j + len("Action Input:") :].strip()
    try:
        json.loads(tool_input)
    except json.JSONDecodeError:
        return content
    tool_calls = []
    tool_call = {
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
    res = qwen_tool_formatter(tools=tools)
    print(res)
    out = 'Action: multiply.\nAction Input: {"first_int": 8, "second_int": 9}\n'
    r = qwen_tool_extractor(out)
    print("\n\n")
    print(r)
