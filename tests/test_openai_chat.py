import openai

# 新版本
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.102.19:8082/v1"
model = "chatglm3"
stream = False
data = {
    "model": model,
    "messages": [{"role": "user", "content": "你是谁"}],
    "stream": stream,
}
completion = openai.ChatCompletion.create(**data)
if stream:
    text = ""
    for choices in completion:
        c = choices.choices[0]
        delta = c.delta
        if hasattr(delta, "content"):
            text += delta.content
            print(delta.content, end="", flush=True)
    print()
else:
    for choice in completion.choices:
        print(choice.message.content)
