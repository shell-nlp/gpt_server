import json
from loguru import logger


def tool_parser(full_text: str, tool_parser, tools, ret):
    tool_call_info = tool_parser.extract_tool_calls(full_text, "")
    tools_called = tool_call_info.tools_called
    text, tool_calls = tool_call_info.content, tool_call_info.tool_calls
    tool_calls = [i.model_dump() for i in tool_calls]
    if tools and tools_called:  # 如果传入tools
        logger.debug(f"工具解析成功, tool_calls: {tool_calls}")
        ret["text"] = text
        ret["tool_calls"] = tool_calls
        ret["finish_reason"] = "tool_calls"
        return json.dumps(ret).encode() + b"\0"
    else:
        ret["text"] = ""
        return json.dumps(ret).encode() + b"\0"
