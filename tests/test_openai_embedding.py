from openai import OpenAI

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
data = client.embeddings.create(model="acge_text_embedding", input=["你是谁", "你是谁"])

print(data.data)
