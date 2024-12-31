from loguru import logger
from typing import Any, Dict, List, Tuple, Union, Optional
import json
import uuid

from gpt_server.model_handler.react.v0.prompt import (
    TOOL_SUFFIX_PROMPT,
)
from gpt_server.model_handler.react.v0.prompts.qwen_prompt import (
    TOOL_SYSTEM_PROMPT_CN,
    TOOl_CHOICE_SYSTEM_PROMPT_CN,
    TOOL_CHOICE_SUFFIX_PROMPT,
)


def qwen_tool_formatter(
    tools: List[Dict[str, Any]], tool_choice_info: Optional[dict] = None
) -> str:
    tool_chooce_suffix_prompt = ""
    logger.info(f"tool_choice_info: {tool_choice_info}")
    tool_system_prompt = TOOL_SYSTEM_PROMPT_CN
    if tool_choice_info:
        tool_chooce_suffix_prompt = TOOL_CHOICE_SUFFIX_PROMPT
        tools = [tools[tool_choice_info["tool_choice_idx"]]]
        logger.info(f"tools 已被替换为tool_choic: {tools}")
        tool_system_prompt = TOOl_CHOICE_SYSTEM_PROMPT_CN

    tool_names = []
    param_text_list = []
    for tool in tools:
        tool = tool["function"]
        tool_name = tool["name"]
        description = tool["description"]
        parameters = tool["parameters"]
        param_text = (
            """### {tool_name}\n\n{tool_name}: {description} 输入参数： {parameters} \n"""
            + TOOL_SUFFIX_PROMPT
            + tool_chooce_suffix_prompt
        )
        param_text_str = param_text.format(
            tool_name=tool_name,
            description=description,
            parameters=parameters,
        )
        param_text_list.append(param_text_str)

        tool_names.append(tool_name)

    tool_text = "\n\n".join(param_text_list).strip()
    return tool_system_prompt.format(
        tool_text=tool_text,
        tool_names=", ".join(tool_names),
    )


def qwen_tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:

    i = content.rfind("Action:")
    j = content.rfind("Action Input:")
    tool_name = content[i + len("Action:") : j].strip().strip(".")
    tool_input = content[j + len("Action Input:") :].strip().split("\n")[0]
    try:
        json.loads(tool_input)
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
    res = qwen_tool_formatter(tools=tools)
    print(res)

#     out = 'Action: multiply.\nAction Input: {"first_int": 8, "second_int": 9}\n'
#     out = """Action: myself
# Action Input: {"question": "你是谁"}
# ✿Retrun✿: 我是通义千问，由阿里云开发的AI助手。我被设计用来回答各种问题、提供信息和与用户进行对话。有什么我可以帮助你的吗？"""
#     r = qwen_tool_extractor(out)
#     print("\n\n")
#     print(r)
