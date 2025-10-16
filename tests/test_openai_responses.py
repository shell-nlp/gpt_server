from openai import OpenAI

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")

stream = True
response = client.responses.create(model="qwen", input="你是谁", stream=False)

print(response)
