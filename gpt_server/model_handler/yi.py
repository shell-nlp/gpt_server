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


def deepspeed_tp():
    """from https://github.com/01-ai/Yi/blob/main/demo/text_generation_tp.py"""
    from deepspeed.module_inject import auto_tp
    import torch.nn as nn

    # module_inject for model Yi
    def is_load_module(module):
        load_layers = [nn.Linear, nn.Embedding, nn.LayerNorm]
        load_layer_names = [
            "LPLayerNorm",
            "SharedEmbedding",
            "OPTLearnedPositionalEmbedding",
            "LlamaRMSNorm",
            # "YiRMSNorm",
        ]
        return module.__class__ in load_layers or module._get_name() in load_layer_names

    auto_tp.Loading.is_load_module = is_load_module

    replace_with_kernel_inject = False
    return replace_with_kernel_inject


if __name__ == "__main__":
    from pprint import pprint

    query, messages = conv2messages(test_prompt)
    print(query)
    pprint(messages, indent=2, sort_dicts=False)
