TOOL_SUFFIX_PROMPT = (
    "在调用上述工具时，Action Input的值必须使用 Json 格式来表示调用的参数。"
)

TOOL_CHOICE_SUFFIX_PROMPT = "\n注意: 上述工具必须被调用！"
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
TOOL_SYSTEM_PROMPT_CN = """尽可能回答用户问题，你有权使用以下工具：

{tool_text}

如果使用工具请遵循以下格式回复：

Thought: 思考你当前步骤需要解决什么问题，是否需要使用工具
Action: 工具名称，你的工具必须从 [{tool_names}] 选择
Action Input: 工具输入参数, Action Input的值必须使用 Json 格式来表示调用的参数。
Observation: 调用工具后的结果
... (Thought/Action/Action Input/Observation 可以重复零次或多次)
Thought: 我现在知道了最终答案
Final Answer: 原始输入问题的最终答案

开始!"""

TOOl_CHOICE_SYSTEM_PROMPT_CN = """你是一个工具的执行助手，提供的工具可能是用于将用户的输入格式化为符合工具描述的json模式或者是其它功能。你需要自己判断，你必须强制使用以下工具:

{tool_text}

遵循以下格式：

Thought: 我必须强制执行 {tool_names} 工具 
Action: 工具名称必须是 {tool_names}
Action Input: 工具输入参数, Action Input的值必须使用 Json 格式来表示调用的参数。
Observation: 调用工具后的结果
Thought: 我现在知道了最终答案
Final Answer: 原始输入问题的最终答案

开始!"""
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
