TOOL_SUFFIX_PROMPT = (
    "在调用上述函数时，Action Input的值必须使用 Json 格式来表示调用的参数。"
)

TOOL_CHOICE_SUFFIX_PROMPT = "\n注意: 上述函数必须被调用！"
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

TOOl_CHOICE_SYSTEM_PROMPT = """You must use the following tools:

{tool_text}

Use the following format:

Question: the input question you must answer
Thought: I have to execute tool {tool_names}
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question:"""
# 你的任务是针对用户的问题和要求提供适当的答复和支持
GLM4_TOOL_PROMPT = """"你可以使用以下工具提供适当的答复和支持。

# 可用工具
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

Question:
"""
