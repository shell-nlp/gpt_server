test_prompt = """
<|system|>
You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
<|user|>
Hello
<|assistant|>
123
<|user|>
Hello
<|assistant|>
123
<|user|>
Hello2
<|assistant|>

"""


# TODO
def conv2messages(prompt):
    # 去除多余的换行符和空格
    prompt = prompt.strip()

    # 将提示模型转换为列表格式
    messages = []
    segments = prompt.split("<|")
    for segment in segments[1:]:
        role, content = segment.split("|>")
        messages.append({"role": role, "content": content.strip()})
    query = None
    for i, item in enumerate(messages):
        if item["role"] == "assistant" and item["content"] == "":
            query = messages[i - 1]["content"]
            messages = messages[: i - 1]
            break
    if len(messages) <= 1:
        messages = []
        return query, messages
    
    history = []
    if messages[0]["role"] == "system":
        system = messages[0]["content"]
        messages = messages[1:]
    for i in range(0, len(messages), 2):
        history.append((messages[i]["content"], messages[i + 1]["content"]))
    if query:
        return query, history
    else:
        raise Exception("conv2messages 解析错误")


if __name__ == "__main__":
    from pprint import pprint

    query, messages = conv2messages(test_prompt)
    print(query)
    pprint(messages, indent=2, sort_dicts=False)
