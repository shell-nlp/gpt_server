import re
from typing import Any, Dict, List, Tuple, Union
import json

# default
TOOL_SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:
{tool_text}
Use the following format if using a tool:
```
Action: tool name (one of [{tool_names}]).
Action Input: the input to the tool{format_prompt}.```"""

JSON_FORMAT_PROMPT = """, in a JSON format representing the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)"""
# GLM4
GLM4_TOOL_SUFFIX_PROMPT = "在调用上述函数时，请使用 Json 格式表示调用的参数。"

GLM4_TOOL_PROMPT = (
    "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持，"
    "{tool_text}"
)


def default_tool_formatter(tools: List[Dict[str, Any]]) -> str:
    tool_text = ""
    tool_names = []
    for tool in tools:
        tool = tool["function"]
        param_text = ""
        for name, param in tool["parameters"]["properties"].items():
            required = (
                ", required" if name in tool["parameters"].get("required", []) else ""
            )
            enum = (
                ", should be one of [{}]".format(", ".join(param["enum"]))
                if param.get("enum", None)
                else ""
            )
            items = (
                ", where each item should be {}".format(param["items"].get("type", ""))
                if param.get("items")
                else ""
            )
            param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                name=name,
                type=param.get("type", ""),
                required=required,
                desc=param.get("description", ""),
                enum=enum,
                items=items,
            )

        tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
            name=tool["name"], desc=tool.get("description", ""), args=param_text
        )
        tool_names.append(tool["name"])

    return TOOL_SYSTEM_PROMPT.format(
        tool_text=tool_text,
        tool_names=", ".join(tool_names),
        format_prompt=JSON_FORMAT_PROMPT,
    )


def glm4_tool_formatter(tools: List[Dict[str, Any]]) -> str:
    tool_text = ""
    for tool in tools:
        tool = tool["function"]
        tool_name = tool["name"]
        tool_text += f"\n\n## {tool_name}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}\n{GLM4_TOOL_SUFFIX_PROMPT}"
    return GLM4_TOOL_PROMPT.format(tool_text=tool_text)


def default_tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:
    regex = re.compile(
        r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*({.*?})(?=\nAction:|\Z)",
        re.DOTALL,
    )
    action_match = re.findall(regex, content)
    if not action_match:
        return content

    results = []

    for match in action_match:
        tool_name, tool_input = match
        tool_name = tool_name.strip()
        tool_input = tool_input.strip().strip('"').strip("```")

        try:
            arguments = json.loads(tool_input)
            results.append((tool_name, json.dumps(arguments, ensure_ascii=False)))
        except json.JSONDecodeError:
            return content

    return results


def glm4_tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:
    lines = content.strip().split("\n")
    if len(lines) != 2:
        return content
    tool_name = lines[0].strip()
    tool_input = lines[1].strip()
    try:
        arguments = json.loads(tool_input)
    except json.JSONDecodeError:
        return content
    return [(tool_name, json.dumps(arguments, ensure_ascii=False))]


def add_tools2messages(params: dict, model_adapter: str = "default"):
    messages = params["messages"]
    if params.get("tools", None):  # 如果传入tools
        if model_adapter == "default":
            system_content = default_tool_formatter(tools=params.get("tools"))

        elif model_adapter == "chatglm4":
            system_content = glm4_tool_formatter(tools=params.get("tools"))

        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_content})

        elif messages[0]["role"] == "system":
            messages[0]["content"] = system_content

    return messages


if __name__ == "__main__":
    import json

    tools_str = """[{'type': 'function', 'function': {'name': 'track', 'description': '追踪指定股票的实时价格', 'parameters': {'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码', 'type': 'integer'}}, 'required': ['symbol']}}}, {'type': 'function', 'function': {'name': 'text-to-speech', 'description': '将文本转换为语音', 'parameters': {'type': 'object', 'properties': {'text': {'description': '需要转换成语音的文本', 'type': 'string'}, 'voice': {'description': '要使用的语音类型（男声、女声等', 'default': '男声', 'type': 'string'}, 'speed': {'description': '语音的速度（快、中等、慢等', 'default': '中等', 'type': 'string'}}, 'required': ['text']}}}]"""
    tools_str = tools_str.replace("'", '"')
    tools = json.loads(tools_str)
    # tools = eval(tools_str)
    print(tools)
    print("#" * 100)
    res = default_tool_formatter(tools=tools)
    print(res)
    # print("#" * 100)
    # res = glm4_tool_formatter(tools=tools)
    # print(res)
