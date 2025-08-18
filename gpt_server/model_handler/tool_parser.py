import json
import uuid
from loguru import logger
import re
from typing import Dict, List, Literal, Sequence, Union, Optional

import partial_json_parser
from pydantic import BaseModel, Field
import shortuuid
from partial_json_parser.core.options import Allow

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)

from lmdeploy.serve.openai.tool_parser import ToolParser, ToolParserManager
from lmdeploy.serve.openai.tool_parser.utils import extract_intermediate_diff
from lmdeploy.serve.openai.tool_parser import ToolParser


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


@ToolParserManager.register_module(["glm"])
class GLMToolParser(ToolParser):

    def __init__(self, tokenizer: object = None):
        super().__init__(tokenizer)
        self.position = 0

    def get_argments(self, obj):
        if "parameters" in obj:
            return obj.get("parameters")
        elif "arguments" in obj:
            return obj.get("arguments")
        return None

    def extract_tool_calls(
        self,
        model_output: str,
        tools,
    ) -> ExtractedToolCallInformation:
        text = model_output
        try:
            if "Action:" not in text:
                raise Exception
            i = text.rfind("Action:")
            j = text.rfind("Action Input:")
            name = text[i + len("Action:") : j].strip().strip(".")
            if "Observation" in model_output:
                k = text.rfind("Observation")
                arguments = text[j + len("Action Input:") : k].strip()
            else:
                arguments = text[j + len("Action Input:") :].strip()
            tool_calls = []
            tool_calls.append(
                ToolCall(function=FunctionCall(name=name, arguments=arguments))
            )
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=text
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=text if len(text) > 0 else "",
        )


@ToolParserManager.register_module(["qwen2_5"])
class Qwen2d5ToolParser(ToolParser):

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.position = 0
        self.tool_start_token = "<tool_call>"
        self.tool_end_token = "</tool_call>"
        self.pattern = r"<tool_call>(.*?)</tool_call>"

    def get_argments(self, obj):
        if "parameters" in obj:
            return obj.get("parameters")
        elif "arguments" in obj:
            return obj.get("arguments")
        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        if self.tool_start_token not in current_text:
            self.position = len(current_text)
            return DeltaMessage(content=delta_text)
        # if the tool call is sended, return a empty delta message
        # to make sure the finish_reason will be send correctly.
        if self.current_tool_id > 0:
            return DeltaMessage(content="")

        last_pos = self.position
        if self.tool_start_token not in current_text[last_pos:]:
            return None

        new_delta = current_text[last_pos:]
        text, action = new_delta.split(self.tool_start_token)

        if len(text) > 0:
            self.position = self.position + len(text)
            return DeltaMessage(content=text)

        action = action.strip()
        action = action.split(self.tool_end_token.strip())[0]

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            parsable_arr = action

            # tool calls are generated in an object in inernlm2
            # it's not support parallel tool calls
            try:
                tool_call_arr: Dict = partial_json_parser.loads(parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
                function_name = tool_call_arr.get("name")
                if function_name:
                    self.current_tool_id = self.current_tool_id + 1
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=f"chatcmpl-tool-{shortuuid.random()}",
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                    self.streamed_args_for_tool.append("")
                else:
                    delta = None
            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                prev_arguments = self.get_argments(
                    self.prev_tool_call_arr[self.current_tool_id]
                )
                cur_arguments = self.get_argments(tool_call_arr)

                # not arguments generated
                if not cur_arguments and not prev_arguments:
                    delta = None
                # will never happen
                elif not cur_arguments and prev_arguments:
                    logger.error(
                        "INVARIANT - impossible to have arguments reset "
                        "mid-arguments"
                    )
                    delta = None
                # first time to get parameters
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)

                    arguments_delta = cur_arguments_json[
                        : cur_arguments_json.index(delta_text) + len(delta_text)
                    ]
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=arguments_delta
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                # both prev and cur parameters, send the increase parameters
                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)

                    argument_diff = extract_intermediate_diff(
                        cur_args_json, prev_args_json
                    )

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=argument_diff
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            tool_call_arr["arguments"] = self.get_argments(tool_call_arr)
            self.prev_tool_call_arr = [tool_call_arr]
            return delta
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction " "error"
            )
            return None

    def extract_tool_calls(
        self,
        model_output: str,
        tools,
    ) -> ExtractedToolCallInformation:
        text = model_output
        if self.tool_start_token in text and self.tool_end_token in text:
            logger.debug("tool_parse tool_start_token 在 text")
            # get tool_call in text
            match_result_list = re.findall(self.pattern, text, re.DOTALL)
            tool_calls = []
            for match_result in match_result_list:
                action = json.loads(match_result)
                name, arguments = action["name"], json.dumps(
                    action["arguments"], ensure_ascii=False
                )
                tool_calls.append(
                    ToolCall(function=FunctionCall(name=name, arguments=arguments))
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
            for match_result in match_result_list:
                index += 1
                action = json.loads(match_result)
                name, arguments = action["name"], json.dumps(
                    action["arguments"], ensure_ascii=False
                )
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


def tool_parser(full_text: str, tool_parser: ToolParser, tools, ret):
    tool_call_info = tool_parser.extract_tool_calls(full_text, tools)
    tools_called = tool_call_info.tools_called
    text, tool_calls = tool_call_info.content, tool_call_info.tool_calls
    tool_calls = [i.model_dump() for i in tool_calls]
    if tools and tools_called:  # 如果传入tools
        logger.info(f"工具解析成功, tool_calls: {tool_calls}")
        ret["text"] = ""
        ret["tool_calls"] = tool_calls
        ret["finish_reason"] = "tool_calls"
        return json.dumps(ret).encode() + b"\0"
    else:
        logger.info(f"工具解析失败, tool_calls: {tool_calls}")
        ret["text"] = ""
        ret["tool_calls"] = tool_calls
        ret["finish_reason"] = "tool_calls"
        return json.dumps(ret).encode() + b"\0"


if __name__ == "__main__":
    full_text = """Action: get_weather
Action Input: {"location": "Nanjing", "unit": "celsius"}"""
    tool_parser2 = ToolParserManager.module_dict["glm"]()
    tool_parser(full_text=full_text, tool_parser=tool_parser2, tools=True, ret={})
