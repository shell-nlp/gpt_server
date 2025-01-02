TOOL_SUFFIX_PROMPT = (
    "在调用上述工具时，action_input的值必须使用 Json 格式来表示调用的参数。"
)

TOOL_CHOICE_SUFFIX_PROMPT = "\n\n## 注意: \n上述工具必须被调用!"
# default

TOOL_SYSTEM_PROMPT_CN = """# 工具
## 你拥有如下工具：

{tool_text}

## 如果使用工具，你可以回复零次、一次或多次以下json格式内容，以调用工具,调用工具后,Observation 表示调用工具后的结果,json格式如下:
{{
    "thought":"你应该时刻思考自己该做什么",
    "reason":{{
        "action":"工具名称，必须是 [{tool_names}] 之一",
        "action_input":"工具输入, 值必须使用 json 格式"
    }}
}}
或
{{
    "thought":"你应该时刻思考自己该做什么",
    "reason":{{
        "final_answer":"根据工具结果进行回复，如果工具返回值存在图片url,需将图片用![](url)渲染出来"
    }}
}}
"""

TOOl_CHOICE_SYSTEM_PROMPT_CN = """# 提供的工具是用于将用户的输入或回复格式化为符合工具描述的json模式,你必须强制使用以下工具:
## 工具
## #你拥有如下工具：

{tool_text}

### 你可以在回复中插入零次、一次或多次以下json格式内容，以调用工具,调用工具后,Observation 表示调用工具后的结果,json格式如下:
{{
    "thought":"你应该时刻思考自己该做什么",
    "reason":{{
        "action":"工具名称，必须是 [{tool_names}] 之一",
        "action_input":"工具输入, 值必须使用 json 格式"
    }}
}}
或
{{
    "thought":"你应该时刻思考自己该做什么",
    "reason":{{
        "final_answer":"根据工具结果进行回复，如果工具返回值存在图片url,需将图片用![](url)渲染出来"
    }}
}}"""
