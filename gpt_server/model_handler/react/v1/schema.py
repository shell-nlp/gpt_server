from pydantic import BaseModel, Field
from typing import Union


class Action(BaseModel):
    Action: str = Field(description="工具名称，必须是 [{tool_names}] 之一")
    Action_Input: str = Field(description="工具输入, 值必须使用 json 格式")


class Answer(BaseModel):
    Final_Answer: str = Field(description="问题的最终回答")


class React(BaseModel):
    Thought: str = Field(description="你应该时刻思考自己该做什么")
    Reason: Union[Action, Answer]
