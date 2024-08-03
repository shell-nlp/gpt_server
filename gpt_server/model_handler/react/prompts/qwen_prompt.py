TOOL_SUFFIX_PROMPT = (
    "在调用上述工具时，Action Input的值必须使用 Json 格式来表示调用的参数。"
)

TOOL_CHOICE_SUFFIX_PROMPT = "\n\n## 注意: \n上述工具必须被调用！"
# default

TOOL_SYSTEM_PROMPT_CN = """# 工具
## 你拥有如下工具：

{tool_text}

## 如果使用工具，你可以在回复中插入零次、一次或多次以下命令以调用工具：

Action: 工具名称，必须是 [{tool_names}] 之一
Action Input: 工具输入, 值必须使用 json 格式
Observation: 调用工具后的结果
✿Retrun✿: 根据工具结果进行回复，需将图片用![](url)渲染出来"""

TOOl_CHOICE_SYSTEM_PROMPT_CN = """你是一个工具的执行助手，提供的工具可能是用于将用户的输入格式化为符合工具描述的json模式或者是其它的功能，你需要自己判断，你必须强制使用以下工具:
# 工具
## 你拥有如下工具：

{tool_text}

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

Thought: 我必须强制执行 {tool_names} 工具 
Action: 工具名称必须是 {tool_names}
Action Input: 工具输入, 值必须使用 json 格式
Observation: 调用工具后的结果
✿Retrun✿: 根据工具结果进行回复，需将图片用![](url)渲染出来"""
