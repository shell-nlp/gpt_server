from gpt_server.model_handler.qwen_react import qwen_tool_formatter
from gpt_server.model_handler.chatglm_react import glm4_tool_formatter


def add_tools2messages(params: dict, model_adapter: str = "default"):
    messages = params["messages"]
    if params.get("tools", None):  # 如果传入tools
        if model_adapter == "qwen":
            system_content = qwen_tool_formatter(tools=params.get("tools"))

        elif model_adapter == "chatglm4":
            system_content = glm4_tool_formatter(tools=params.get("tools"))

        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_content})

        elif messages[0]["role"] == "system":
            messages[0]["content"] = system_content

    return messages
