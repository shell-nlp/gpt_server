from pydantic import BaseModel, Field
from typing import Union


class Action(BaseModel):
    action: str = Field(description="工具名称，必须是 [{tool_names}] 之一")
    action_input: str = Field(description="工具输入, 值必须使用 json 格式")


class Answer(BaseModel):
    final_answer: str = Field(description="问题的最终回答")


class React(BaseModel):
    thought: str = Field(description="你应该时刻思考自己该做什么")
    reason: Union[Action, Answer]
