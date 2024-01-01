test_prompt = """

<|user|>
Hello
<|assistant|>
1
<|user|>
Hello
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

    if query:
        messages.append({"role": "user", "content": query})
        return messages
    else:
        raise Exception("conv2messages 解析错误")


if __name__ == "__main__":
    from pprint import pprint

    query, messages = conv2messages(test_prompt)
    print(query)
    pprint(messages, indent=2, sort_dicts=False)
