from gpt_server.model_handler.qwen_react import qwen_tool_formatter
from gpt_server.model_handler.chatglm_react import glm4_tool_formatter
from loguru import logger


def add_tools2messages(params: dict, model_adapter: str = "default"):
    messages = params["messages"]
    tools = params.get("tools", None)
    # {'type': 'function', 'function': {'name': 'AnswerWithJustification'}}
    tool_choice = params.get("tool_choice", None)
    tool_choice_info = None
    if tool_choice:
        idx = -1
        for tool in tools:
            idx += 1

            if (
                tool_choice["type"] == tool["type"]
                and tool_choice["function"]["name"] == tool["function"]["name"]
            ):
                tool_choice_info = {"tool_choice_idx": idx}
                logger.info(f"tool_choice执行: {tool_choice_info}")
                break

    if tools:  # 如果传入tools
        if model_adapter == "qwen":
            system_content = qwen_tool_formatter(
                tools=params.get("tools"), tool_choice_info=tool_choice_info
            )

        elif model_adapter == "chatglm4":
            system_content = glm4_tool_formatter(
                tools=params.get("tools"), tool_choice_info=tool_choice_info
            )

        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_content})

        elif messages[0]["role"] == "system":
            messages[0]["content"] = system_content

    return messages
