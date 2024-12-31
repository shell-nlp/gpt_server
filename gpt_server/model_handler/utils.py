from gpt_server.model_handler.react.v0.qwen_react import qwen_tool_formatter
from gpt_server.model_handler.react.v0.chatglm_react import glm4_tool_formatter
from gpt_server.model_handler.react.v0.system_react import system_tool_formatter
from loguru import logger
from typing import Literal, Optional, Union


def formatter_messages(
    messages: dict, model_adapter: str, tool_choice_info, params: dict
):
    # 根据模型适配器获取系统消息内容
    if model_adapter == "qwen":
        system_content = qwen_tool_formatter(
            tools=params.get("tools"), tool_choice_info=tool_choice_info
        )

    elif model_adapter == "chatglm4":
        system_content = glm4_tool_formatter(
            tools=params.get("tools"), tool_choice_info=tool_choice_info
        )
    else:
        system_content = system_tool_formatter(
            tools=params.get("tools"), tool_choice_info=tool_choice_info
        )
    # 在消息列表中插入或更新系统消息
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_content})

    elif messages[0]["role"] == "system":
        messages[0]["content"] = messages[0]["content"] + "\n\n" + system_content
    return messages


def add_tools2messages(
    params: dict,
    model_adapter: Optional[
        Union[Literal["default"], Literal["qwen"], Literal["chatglm4"]]
    ] = "default",
):
    """根据工具选择和适配器添加工具信息到消息"""
    messages = params.get("messages", [])
    # ------------------------------------------------
    tools = params.get("tools", None)
    tool_choice = params.get("tool_choice", "none")
    # -----------------------------
    if tool_choice == "none":  # none 表示模型将不调用任何工具，而是生成一条消息。
        return messages
    elif (
        tool_choice == "auto"
    ):  # auto 表示模型可以在生成消息或调用一个或多个工具之间进行选择。
        if tools:  # 如果传入tools
            return formatter_messages(
                messages=messages,
                model_adapter=model_adapter,
                tool_choice_info=None,
                params=params,
            )
        else:
            return messages

    elif tool_choice == "required":  # required  表示模型必须调用一个或多个工具。
        if tools:
            raise Exception("tool_choice 暂时不支持 required")
        else:
            raise Exception("tool_choice == required 时,必须设置tools参数")
    elif isinstance(
        tool_choice, dict
    ):  # {'type': 'function', 'function': {'name': 'AnswerWithJustification'}} 会强制模型调用该工具
        # if tool_choice:
        """首先，要强制执行的工具，必须在 tools 中"""
        idx = 0
        for tool in tools:

            if (
                tool_choice["type"] == tool["type"]
                and tool_choice["function"]["name"] == tool["function"]["name"]
            ):
                tool_choice_info = {"tool_choice_idx": idx}
                logger.info(f"tool_choice执行: {tool_choice_info}")
                break
            idx += 1
        if idx == len(tools):  # 说明工具没有在tools 中
            raise Exception("设置的 tool_choice 在 tools 中不存在！")
        if tools:  # 如果传入tools
            return formatter_messages(
                messages=messages,
                model_adapter=model_adapter,
                tool_choice_info=tool_choice_info,
                params=params,
            )
    return messages
