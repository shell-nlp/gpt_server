from openai import OpenAI

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")

stream = True
input_ = "你是谁"
input_ = [{"role": "user", "content": "你是谁"}]
response = client.responses.create(model="qwen", input=input_, stream=stream)


if stream:
    for event in response:
        print(event)
else:
    print(response, end="\n\n")
